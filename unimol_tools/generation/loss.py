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