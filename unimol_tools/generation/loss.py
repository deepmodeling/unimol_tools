import torch
import torch.nn as nn
import torch.nn.functional as F

class VAELoss(nn.Module):
    def __init__(self, beta=1.0, pad_idx=None):
        super().__init__()
        self.beta = beta
        self.pad_idx = pad_idx
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="mean")

    def forward(self, logits, targets, mean, logv):
        # Reconstruction loss (Cross Entropy)
        # logits: [batch_size, seq_len, vocab_size]
        # targets: [batch_size, seq_len]
        
        vocab_size = logits.size(-1)
        recon_loss = self.criterion(logits.view(-1, vocab_size), targets.reshape(-1))
        
        # KL Divergence
        # KL(q(z|x) || p(z)) where p(z) ~ N(0, 1)
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp(), dim=1)
        kl_loss = torch.mean(kl_loss)
        
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
