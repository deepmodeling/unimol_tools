from dataclasses import dataclass, field
from typing import Optional

@dataclass
class GenerationConfig:
    data_path: str = field(
        default="",
        metadata={"help": "Path to the training dataset."},
    )
    dict_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to the dictionary file."},
    )
    vae_dict_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to the VAE dictionary file."},
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for training."},
    )
    max_epochs: int = field(
        default=10,
        metadata={"help": "Maximum number of training epochs."},
    )
    lr: float = field(
        default=1e-4,
        metadata={"help": "Learning rate."},
    )
    beta: float = field(
        default=0.001,
        metadata={"help": "Weight for KL divergence loss (beta-VAE)."},
    )
    latent_dim: int = field(
        default=256,
        metadata={"help": "Dimension of the latent space."},
    )
    encoder_layers: int = field(
        default=6,
        metadata={"help": "Number of encoder layers."},
    )
    decoder_layers: int = field(
        default=6,
        metadata={"help": "Number of decoder layers."},
    )
    encoder_embed_dim: int = field(
        default=512,
        metadata={"help": "Encoder embedding dimension."},
    )
    decoder_embed_dim: int = field(
        default=512,
        metadata={"help": "Decoder embedding dimension."},
    )
    encoder_attention_heads: int = field(
        default=8,
        metadata={"help": "Number of encoder attention heads."},
    )
    decoder_attention_heads: int = field(
        default=8,
        metadata={"help": "Number of decoder attention heads."},
    )
    encoder_ffn_embed_dim: int = field(
        default=2048,
        metadata={"help": "Encoder FFN embedding dimension."},
    )
    decoder_ffn_embed_dim: int = field(
        default=2048,
        metadata={"help": "Decoder FFN embedding dimension."},
    )
    activation_fn: str = field(
        default="gelu",
        metadata={"help": "Activation function."},
    )
    dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout probability."},
    )
    max_seq_len: int = field(
        default=256,
        metadata={"help": "Maximum sequence length."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."},
    )
    output_dir: str = field(
        default="checkpoints_generation",
        metadata={"help": "Directory to save checkpoints."},
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "Use mixed precision training."},
    )
    # Fields required by UniMolModel (pretrain)
    masked_token_loss: float = field(default=0.0, metadata={"help": "Masked token loss weight."})
    masked_coord_loss: float = field(default=0.0, metadata={"help": "Masked coord loss weight."})
    masked_dist_loss: float = field(default=0.0, metadata={"help": "Masked dist loss weight."})
    emb_dropout: float = field(default=0.1, metadata={"help": "Embedding dropout."})
    attention_dropout: float = field(default=0.1, metadata={"help": "Attention dropout."})
    activation_dropout: float = field(default=0.0, metadata={"help": "Activation dropout."})
    delta_pair_repr_norm_loss: float = field(default=-1.0, metadata={"help": "Delta pair repr norm loss."})
    warmup_steps: int = field(default=10000, metadata={"help": "Warmup steps."}) # Increased warmup
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay."})
