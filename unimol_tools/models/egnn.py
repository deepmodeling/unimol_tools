import torch
import torch.nn as nn
import torch.nn.functional as F

class E_GCL(nn.Module):
    """等变图卷积层：同时更新节点特征 h 和 坐标 x"""
    def __init__(self, input_nf, output_nf, hidden_nf, act_fn=nn.SiLU()):
        super(E_GCL, self).__init__()
        # Edge MLP: 输入 (h_i, h_j, radial_dist_sq)
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_nf * 2 + 1, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )

        # Node MLP: 输入 (h_i, agg_m)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf)
        )

        # Coordinate MLP: 输入 m_ij，输出一个标量权重
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1, bias=False),
            nn.Tanh()
        )

        self.layer_norm = nn.LayerNorm(output_nf)
        nn.init.xavier_uniform_(self.coord_mlp[-2].weight, gain=0.001)

    def forward(self, h, x, mask=None):
        """
        h: [B, N, D]
        x: [B, N, 3]
        mask: [B, N] (1 for atom, 0 for pad)
        """
        batch_size, n_nodes, _ = x.shape
        
        # 1. 构造全连接图的索引 (i, j)
        # 这里为了效率使用广播机制处理全连接
        # h_i: [B, N, 1, D], h_j: [B, 1, N, D]
        h_i = h.unsqueeze(2).repeat(1, 1, n_nodes, 1)
        h_j = h.unsqueeze(1).repeat(1, n_nodes, 1, 1)
        
        # 相对坐标和平方距离
        # rel_x: [B, N, N, 3]
        rel_x = x.unsqueeze(2) - x.unsqueeze(1)
        dist_sq = torch.sum(rel_x**2, dim=-1, keepdim=True) + 1e-8 # [B, N, N, 1]

        # 2. Edge Messaging
        edge_input = torch.cat([h_i, h_j, dist_sq], dim=-1) # [B, N, N, 2D+1]
        m_ij = self.edge_mlp(edge_input) # [B, N, N, hidden]
        
        if mask is not None:
            # 屏蔽 padding 原子的消息
            combined_mask = mask.unsqueeze(2) * mask.unsqueeze(1) # [B, N, N]
            m_ij = m_ij * combined_mask.unsqueeze(-1)

        # 3. Coordinate Update (Equivariant)
        # x_i = x_i + sum_j (x_i - x_j) * phi_x(m_ij)
        # weight: [B, N, N, 1]
        w_ij = self.coord_mlp(m_ij) * 0.1
        if mask is not None:
            w_ij = w_ij * combined_mask.unsqueeze(-1)
            
        # 这里的 1/N 是为了稳定性，防止坐标爆炸
        coord_diff = torch.sum(rel_x * w_ij, dim=2) / n_nodes 
        x = x + coord_diff

        # 4. Node Update (Invariant)
        agg_m = torch.sum(m_ij, dim=2) # [B, N, hidden]
        h_new = self.node_mlp(torch.cat([h, agg_m], dim=-1))
        h = h + h_new # 残差连接
        h = self.layer_norm(h)

        return h, x

class EGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            E_GCL(in_channels if i==0 else hidden_channels, 
                  hidden_channels, hidden_channels) 
            for i in range(num_layers)
        ])

    def forward(self, h, x, mask=None):
        for layer in self.layers:
            h, x = layer(h, x, mask=mask)
        return h, x