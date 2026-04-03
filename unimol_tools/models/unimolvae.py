import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import math
from ..pretrain.unimol import UniMolModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq_len, batch, dim]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class UniMolVAE(nn.Module):
    def __init__(self, config, dictionary, vae_dict):
        super().__init__()
        self.config = config
        self.dictionary = dictionary
        self.vae_dict = vae_dict
        self.padding_idx = dictionary.pad()
        self.target_padding_idx = self.vae_dict.pad()
        self.vocab_size = len(dictionary)
        self.target_vocab_size = len(self.vae_dict)
        
        self.check_config(config.model)

        # Encoder: use UniMolModel (pre-trained architecture)
        # Note: UniMolModel expects the model config which holds model hyperparameters
        self.unimol_encoder = UniMolModel(config.model, dictionary)
        
        # Variational parameters
        self.fc_mu = nn.Linear(config.model.encoder_embed_dim, config.model.latent_dim)
        self.fc_var = nn.Linear(config.model.encoder_embed_dim, config.model.latent_dim)
        
        # Decoder Input Projection (Latent Z -> Decoder Memory)
        self.k = config.model.latent_slots  # 建议=8或16

        self.z_to_memory = nn.Linear(
            config.model.latent_dim,
            self.k * config.model.decoder_embed_dim
        )

        # ⭐ embedding conditioning（强烈建议）
        self.z_to_embed = nn.Linear(
            config.model.latent_dim,
            config.model.decoder_embed_dim
        )

        # Decoder Embedding (SMILES tokens)
        self.decoder_embed_tokens = nn.Embedding(self.target_vocab_size, config.model.decoder_embed_dim, padding_idx=self.target_padding_idx)
        self.pos_encoder = PositionalEncoding(config.model.decoder_embed_dim, config.model.dropout)
        
        decoder_layers = TransformerDecoderLayer(
            d_model=config.model.decoder_embed_dim, 
            nhead=config.model.decoder_attention_heads, 
            dim_feedforward=config.model.decoder_ffn_embed_dim, 
            dropout=config.model.dropout, 
            activation=config.model.activation_fn
        )
        self.decoder = TransformerDecoder(decoder_layers, config.model.decoder_layers)
        
        self.fc_out = nn.Linear(config.model.decoder_embed_dim, self.target_vocab_size)


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

    def check_config(self, config):
        required = ['encoder_embed_dim', 'decoder_embed_dim', 'latent_dim', 
                    'decoder_layers', 'decoder_attention_heads', 'decoder_ffn_embed_dim', 
                    'dropout', 'activation_fn']
        for k in required:
            if not hasattr(config, k):
                # Pass if strict check not strictly desired or set defaults
                pass
                # raise ValueError(f"Missing config attribute: {k}")

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src_tokens, src_distance, src_coord, src_edge_type, decoder_input_tokens=None, **kwargs):
        # Encoder Forward
        # UniMolModel.forward returns: (logits, encoder_distance, encoder_coord, x_norm, delta_encoder_pair_rep_norm)
        # We need logits as representation.
        encoder_rep, _, _, _, _ = self.unimol_encoder(
            src_tokens, 
            src_distance, 
            src_coord, 
            src_edge_type
        )
        
        # Take [CLS] token representation (index 0)
        # encoder_rep: [batch, seq_len, dim] usually for UniMol
        hidden = encoder_rep[:, 0, :] 
        
        mu = self.fc_mu(hidden)
        logvar = self.fc_var(hidden)
        
        z = self.reparameterize(mu, logvar)
        
        if decoder_input_tokens is None:
            # Inference mode requires explicit generation loop, not handled here
            return {"z": z, "mean": mu, "logv": logvar}
            
        B = z.size(0)
        D = self.config.model.decoder_embed_dim
        k = self.k

        z_mem = self.z_to_memory(z)             # [B, k*D]
        z_mem = z_mem.view(B, k, D)             # [B, k, D]
        z_mem = z_mem.transpose(0, 1)           # [k, B, D]

        # Decoder Input: decoder_input_tokens (SMILES tokens)
        tgt_emb = self.decoder_embed_tokens(decoder_input_tokens) * math.sqrt(D)
        z_embed = self.z_to_embed(z).unsqueeze(1)  # [B, 1, D]
        tgt_emb = tgt_emb + z_embed

        tgt_emb = tgt_emb.transpose(0, 1) # [seq_len, batch, dim]
        tgt_emb = self.pos_encoder(tgt_emb)
        
        decoder_input_tokens = random_mask(decoder_input_tokens, self.target_padding_idx, mask_prob=0.15)
        tgt_key_padding_mask = (decoder_input_tokens == self.target_padding_idx)
        
        sz = tgt_emb.size(0)
        tgt_mask = self.generate_square_subsequent_mask(sz).to(tgt_emb.device)
        
        output = self.decoder(tgt_emb, z_mem, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        output = output.transpose(0, 1) # [batch, seq_len, dim]
        logits = self.fc_out(output)
        
        return {
            "logits": logits,
            "mean": mu,
            "logv": logvar,
            "z": z
        }
    
def random_mask(tokens, pad_idx, mask_prob=0.1):
    mask = (torch.rand_like(tokens.float()) < mask_prob) & (tokens != pad_idx)
    tokens = tokens.clone()
    tokens[mask] = pad_idx
    return tokens
