import torch
import torch.nn as nn
import torch.nn.functional as F

class VAELoss(nn.Module):
    def __init__(self, beta=1.0, pad_idx=None, token_weights=None):
        super().__init__()
        self.beta = beta
        self.pad_idx = pad_idx

        self.criterion = nn.CrossEntropyLoss(
            weight=token_weights,
            ignore_index=pad_idx,
            reduction="mean"
        )

    def forward(self, logits, targets, mean, logv, step=None):
        vocab_size = logits.size(-1)

        recon_loss = self.criterion(
            logits.view(-1, vocab_size),
            targets.reshape(-1)
        )

        kl_loss = -0.5 * torch.sum(
            1 + logv - mean.pow(2) - logv.exp(),
            dim=1
        )

        # ⭐ free bits（防 collapse）
        # kl_loss = torch.clamp(kl_loss, min=0.5)

        kl_loss = torch.mean(kl_loss)

        if step is None:
            beta = self.beta
        else:
            beta = self.beta * (step / 50000)  # 线性 warm-up
            beta = min(beta, self.beta)

        total_loss = recon_loss + beta * kl_loss

        return total_loss, recon_loss, kl_loss

def get_token_weights(encoder_dict, vae_dict):
    """
    根据 token 类型自动生成权重

    Args:
        encoder_dict: 原始字典（元素 + 特殊token）
        vae_dict: 最终VAE字典（list 或 Dictionary）

    Returns:
        torch.Tensor: [vocab_size]
    """

    # -------- 获取 token 列表 --------
    if hasattr(vae_dict, "symbols"):
        tokens = list(vae_dict.symbols)
    else:
        tokens = list(vae_dict)

    if hasattr(encoder_dict, "symbols"):
        encoder_tokens = set(encoder_dict.symbols)
    else:
        encoder_tokens = set(encoder_dict)

    vocab_size = len(tokens)
    weights = torch.ones(vocab_size)

    token2id = {t: i for i, t in enumerate(tokens)}

    # -------- 定义分类 --------
    strong_structure = {'[', ']'}
    branch_tokens = {'(', ')', '.', ':'}
    bond_tokens = {'=', '#', '-', '/', '\\'}
    stereo_tokens = {'@', '@@'}
    digit_tokens = set(str(i) for i in range(10))

    # 动态识别
    def is_charge(token):
        return token.startswith('+') or token.startswith('-') and len(token) > 1

    def is_hcount(token):
        return token.startswith('H') and len(token) > 1

    # -------- 分配权重 --------
    for token, idx in token2id.items():

        # 1️⃣ encoder token（原子 + special）
        if token in encoder_tokens:
            weights[idx] = 1.3

        # 2️⃣ 强结构
        elif token in strong_structure:
            weights[idx] = 2.5

        # 3️⃣ 分支
        elif token in branch_tokens:
            weights[idx] = 2.0

        # 4️⃣ 键
        elif token in bond_tokens:
            weights[idx] = 1.8

        # 5️⃣ 手性
        elif token in stereo_tokens:
            weights[idx] = 1.5

        # 6️⃣ 数字（环）
        elif token in digit_tokens:
            weights[idx] = 1.5

        # 7️⃣ 电荷
        elif is_charge(token):
            weights[idx] = 1.4

        # 8️⃣ H count
        elif is_hcount(token):
            weights[idx] = 1.3

        # 9️⃣ 其他
        else:
            weights[idx] = 1.0

    return weights


class EDMLoss(nn.Module):
    def __init__(
        self,
        sigma_min=0.002,
        sigma_max=80,
        rho=7,
        lambda_z=0.0,
        lambda_dist=0.0,
        pad_idx=None
    ):
        super().__init__()

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

        self.lambda_z = lambda_z
        self.lambda_dist = lambda_dist

        self.pad_idx = pad_idx

        self.ce = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="mean")

    # ===== sigma sampling（Karras EDM）=====
    def sample_sigma(self, B, device):
        u = torch.rand(B, 1, device=device)
        sigma = (
            self.sigma_max ** (1 / self.rho)
            + u * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho
        return sigma

    def forward(self, model, batch):
        """
        Correct EDM forward for Uni-Mol
        """

        net_input = batch["net_input"]

        tokens = net_input["src_tokens"]        # [B, N]
        x0 = net_input["src_coord"]             # [B, N, 3]
        src_edge_type = net_input["src_edge_type"]

        mask = (tokens != self.pad_idx)         # [B, N]

        B = x0.size(0)
        device = x0.device

        # ===== 1. center x0 =====
        x0_masked = x0 * mask.unsqueeze(-1)
        x0_mean = x0_masked.sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
        x0 = (x0 - x0_mean) * mask.unsqueeze(-1)

        # ===== 2. sample sigma =====
        sigma = self.sample_sigma(B, device)

        # ===== 3. add noise =====
        noise = torch.randn_like(x0)
        x_sigma = x0 + sigma.view(B,1,1) * noise

        # ===== 4. re-center x_sigma =====
        x_sigma_masked = x_sigma * mask.unsqueeze(-1)
        x_sigma_mean = x_sigma_masked.sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
        x_sigma = (x_sigma - x_sigma_mean) * mask.unsqueeze(-1)

        # ===== 5. recompute distance（关键修正）=====
        diff = x_sigma.unsqueeze(2) - x_sigma.unsqueeze(1)   # [B,N,N,3]
        src_distance_sigma = torch.norm(diff, dim=-1)        # [B,N,N]

        # ===== 6. forward =====
        dx, atom_logits = model(
            tokens,
            src_distance_sigma,   # ✅ 用 noisy distance
            x_sigma,
            src_edge_type,
            sigma
        )

        # ===== 7. EDM reconstruction =====
        c_skip = 1 / (sigma**2 + 1)
        c_out  = sigma / torch.sqrt(sigma**2 + 1)

        pred_x0 = (
            c_skip.view(B,1,1) * x_sigma
            + c_out.view(B,1,1) * dx
        )

        # ===== 8. coordinate loss =====
        loss_x = (((pred_x0 - x0) ** 2) * mask.unsqueeze(-1)).sum() / mask.sum().clamp(min=1)

        # ===== 9. atom loss =====
        if self.lambda_z > 0:
            loss_z = self.ce(
                atom_logits.view(-1, atom_logits.size(-1)),
                tokens.view(-1)
            )
        else:
            loss_z = torch.tensor(0.0, device=device)

        # ===== 10. distance loss =====
        if self.lambda_dist > 0:
            dist_pred = torch.cdist(pred_x0, pred_x0)
            dist_gt   = torch.cdist(x0, x0)

            mask_2d = mask.unsqueeze(1) & mask.unsqueeze(2)

            loss_dist = (
                ((dist_pred - dist_gt).abs() * mask_2d).sum()
                / mask_2d.sum().clamp(min=1)
            )
        else:
            loss_dist = torch.tensor(0.0, device=device)

        # ===== 11. total =====
        total_loss = (
            loss_x
            + self.lambda_z * loss_z
            + self.lambda_dist * loss_dist
        )

        return total_loss, loss_x, loss_z, loss_dist