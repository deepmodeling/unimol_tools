import torch
import torch.nn as nn
from ..pretrain.unimol import UniMolModel
from .egnn import EGNN
    
class UniMolSE3EDM(nn.Module):
    def __init__(self, config, dictionary):
        super().__init__()
        self.config = config
        self.dictionary = dictionary
        self.padding_idx = dictionary.pad()

        D = config.model.encoder_embed_dim
        num_atom_types = len(dictionary)

        # ===== Uni-Mol encoder =====
        self.unimol_encoder = UniMolModel(config.model, dictionary)

        # ===== sigma embedding =====
        self.sigma_embed = nn.Sequential(
            nn.Linear(1, D),
            nn.SiLU(),
            nn.Linear(D, D)
        )

        # ===== atom type head =====
        self.atom_head = nn.Sequential(
            nn.Linear(D, D),
            nn.SiLU(),
            nn.Linear(D, num_atom_types)
        )

        # ===== atom embedding（用于condition）=====
        self.atom_embedding = nn.Embedding(num_atom_types, D)

        # ===== EGNN =======
        self.egnn = EGNN(
            in_channels=D,
            hidden_channels=D,
            num_layers=config.model.egnn_layers,
        )

        self.cls_to_node = nn.Sequential(
            nn.Linear(D, D),
            nn.SiLU(),
            nn.Linear(D, D)
        )

    def load_unimol_weights(self, path):
        if path is not None:
            import os
            if os.path.exists(path):
                state_dict = torch.load(path, map_location='cpu')
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                elif 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                self.unimol_encoder.load_state_dict(state_dict, strict=False)

    def forward(self, src_tokens, src_distance, src_coord, src_edge_type, sigma):
        """
        src_tokens: [B, N]
        src_distance: [B, N, N]
        src_coord: [B, N, 3] (Noisy coordinates)
        sigma: [B, 1]
        """
        # 0. Mask for padding
        mask = src_tokens.ne(self.padding_idx).float() # [B, N]

        # 1. Uni-Mol Encoder 提取结构感知特征
        h, _, _, _, _ = self.unimol_encoder(
            src_tokens, 
            src_distance, 
            src_coord, 
            src_edge_type
        )

        # 2. 提取全局特征（CLS token）
        h_cls = h[:, 0, :] # [B, D]

        # 3. 构造重构所需的节点特征
        node_features = self.atom_embedding(src_tokens) # [B, N, D]

        # 4. 将全局特征注入每个节点
        sigma_emb = self.sigma_embed(torch.log(sigma + 1e-8)).unsqueeze(1) # [B, 1, D]
        global_condition = self.cls_to_node(h_cls).unsqueeze(1)

        h_for_egnn = node_features + sigma_emb + global_condition

        # 5. SE(3) 重构
        h_out, final_x = self.egnn(h_for_egnn, src_coord, mask=mask)

        # 6. 预测原子类型 (Invariant)
        atom_logits = self.atom_head(h_out)

        # 7. 去质心 (Center of Gravity removal) - 保证平移不变性非常关键
        dx = final_x - src_coord

        return dx, atom_logits

