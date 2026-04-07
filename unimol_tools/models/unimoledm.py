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

        # ===== Uni-Mol encoder =====
        self.unimol_encoder = UniMolModel(config.model, dictionary)
        self.unimol_encoder.eval()
        
        self.mask_id = self.unimol_encoder.mask_idx

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
            nn.Linear(D, len(dictionary))
        )

        # ===== atom embedding（用于condition）=====
        self.atom_embedding = nn.Embedding(len(dictionary), D)

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
        self.property_mlp = nn.ModuleDict()

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

    def forward(self, src_tokens, src_distance, src_coord, src_edge_type, sigma, target_property=None):
        """
        src_tokens: [B, N]
        src_distance: [B, N, N]
        src_coord: [B, N, 3] (Noisy coordinates)
        sigma: [B, 1]
        """
        # ==================== 重点修改 1 & 2 ====================
        # 0. Mask for physical atoms (排除 padding, CLS/bos, EOS)
        bool_mask = (src_tokens != self.padding_idx) \
                  & (src_tokens != self.dictionary.bos()) \
                  & (src_tokens != self.dictionary.eos())
        mask = bool_mask.float() # [B, N]

        # 1. 冻结 Uni-Mol Encoder 提取结构感知特征
         # 确保 Dropout/BatchNorm 失效
        with torch.no_grad():
            h, _, _, _, _ = self.unimol_encoder(
                src_tokens, 
                src_distance, 
                src_coord, 
                src_edge_type
            )

        # 2. 提取全局特征（CLS token 往往固定在 index=0，但请确认你的字典，下面沿用 h[:,0,:]）
        # h_cls = h[:, 0, :] # [B, D]
        # ========================================================

        # 3. 构造重构所需的节点特征      
        infill_tokens = torch.full_like(src_tokens, self.mask_id) # [B, N]
        infill_tokens = torch.where(src_tokens == self.padding_idx, self.padding_idx, infill_tokens)
        node_features = self.atom_embedding(infill_tokens) # [B, N, D]

        # 4. 将全局特征注入每个节点，并加上隐层节点表征
        sigma_emb = self.sigma_embed(torch.log(sigma + 1e-8)).unsqueeze(1) # [B, 1, D]

        if target_property is not None:
            property_emb = self.property_mlp(target_property) # [B, D]
            global_condition = self.cls_to_node(property_emb).unsqueeze(1)
        else:
            global_condition = 0 # 无条件生成

        # ✅ 正确融合了局部特征 h
        h_for_egnn = node_features + sigma_emb + global_condition + h

        # 5. SE(3) 重构，此处 mask 排除了 CLS 和 EOS，EGNN 物理域更干净
        h_out, final_x = self.egnn(h_for_egnn, src_coord, mask=mask)

        # 6. 预测原子类型 (Invariant)
        atom_logits = self.atom_head(h_out)

        # ==================== 重点修改 3 ====================
        # 7. 真正的去质心 (Center of Gravity removal) 偏移量校正
        dx = final_x - src_coord
        
        # 计算在所有有效原子节点上的偏移均值
        dx_masked = dx * mask.unsqueeze(-1)
        dx_mean = dx_masked.sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
        
        # 让所有生成的有效原子的运动向量之和为 0，防止整个分子“飘走”
        dx = (dx - dx_mean) * mask.unsqueeze(-1)
        # ========================================================

        return dx, atom_logits

