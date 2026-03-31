from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore

@dataclass
class DatasetConfig:
    train_path: str = field(
        default="",
        metadata={"help": "Path to the training dataset."},
    )
    valid_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the validation dataset (optional)."},
    )
    dict_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to the dictionary file."},
    )
    vae_dict_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to the VAE dictionary file."},
    )

@dataclass
class ModelConfig:
    unimol_weight_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained UniMol weights (optional)."},
    )
    latent_dim: int = field(
        default=256,
        metadata={"help": "Dimension of the latent space."},
    )
    encoder_layers: int = field(
        default=15,
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
        default=64,
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
        default=512,
        metadata={"help": "Maximum sequence length."},
    )
    # Fields required by UniMolModel (pretrain)
    masked_token_loss: float = field(default=0.0, metadata={"help": "Masked token loss weight."})
    masked_coord_loss: float = field(default=0.0, metadata={"help": "Masked coord loss weight."})
    masked_dist_loss: float = field(default=0.0, metadata={"help": "Masked dist loss weight."})
    emb_dropout: float = field(default=0.1, metadata={"help": "Embedding dropout."})
    attention_dropout: float = field(default=0.1, metadata={"help": "Attention dropout."})
    activation_dropout: float = field(default=0.0, metadata={"help": "Activation dropout."})
    pooler_dropout: float = field(default=0.2, metadata={"help": "Pooler dropout."})
    pooler_activation_fn: str = field(default="tanh", metadata={"help": "Pooler activation function."})
    post_ln: bool = field(default=False, metadata={"help": "Post layer norm."})
    backbone: str = field(default="transformer", metadata={"help": "Model backbone."})
    kernel: str = field(default="gaussian", metadata={"help": "Kernel function."})
    delta_pair_repr_norm_loss: float = field(default=-1.0, metadata={"help": "Delta pair repr norm loss."})


@dataclass
class TrainingConfig:
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
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."},
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to save checkpoints."},
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "Use mixed precision training."},
    )
    warmup_steps: int = field(default=10000, metadata={"help": "Warmup steps."}) # Increased warmup
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay."})


@dataclass
class GenerationConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

cs = ConfigStore.instance()
cs.store(name="generation_config", node=GenerationConfig)
