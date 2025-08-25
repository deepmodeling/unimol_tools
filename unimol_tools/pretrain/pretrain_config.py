from dataclasses import dataclass, field
from typing import Optional, Tuple

from hydra.core.config_store import ConfigStore


@dataclass
class DatasetConfig:
    train_path: str = ""
    valid_path: Optional[str] = None
    data_type: str = field(
        default="lmdb",
        metadata={"help": "Dataset format, e.g. 'lmdb', 'csv', 'txt', 'smi', or 'sdf'"},
    )
    smiles_column: str = field(
        default="smi",
        metadata={"help": "Column name for SMILES when reading csv data"},
    )
    dict_path: Optional[str] = None
    remove_hydrogen: bool = False
    max_atoms: int = 256
    noise_type: str = "uniform"
    noise: float = 1.0
    mask_prob: float = 0.15
    leave_unmasked_prob: float = 0.05
    random_token_prob: float = 0.05
    add_2d: bool = True
    num_conformers: int = 10

@dataclass
class ModelConfig:
    model_name: str = 'UniMol'
    encoder_layers: int = 15
    encoder_embed_dim: int = 512
    encoder_ffn_embed_dim: int = 2048
    encoder_attention_heads: int = 64
    dropout: float = 0.1
    emb_dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    pooler_dropout: float = 0.0
    max_seq_len: int = 512
    activation_fn: str = "gelu"
    pooler_activation_fn: str = "tanh"
    post_ln: bool = False
    masked_token_loss: float = 1
    masked_coord_loss: float = 5
    masked_dist_loss: float = 10
    x_norm_loss: float = 0.01
    delta_pair_repr_norm_loss: float = 0.01

@dataclass
class TrainingConfig:
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    adam_betas: Tuple[float, float] = (0.9, 0.99)
    adam_eps: float = 1e-6
    clip_grad_norm: float = 1.0
    epochs: int = field(
        default=0,
        metadata={"help": "Number of epochs; 0 to disable and rely on total_steps"},
    )
    total_steps: int = field(
        default=10000, metadata={"help": "Maximum number of training steps"}
    )
    warmup_steps: int = field(
        default=1000,
        metadata={"help": "Number of steps to linearly warm up the learning rate"},
    )
    log_every_n_steps: int = 10
    save_every_n_steps: int = 100
    keep_last_n_checkpoints: int = field(
        default=3,
        metadata={"help": "How many step checkpoints to keep"},
    )
    patience: int = field(
        default=-1,
        metadata={"help": "Early stop patience based on validation loss; -1 disables"},
    )
    use_amp: bool = True
    local_rank: int = 0
    seed: int = 42
    resume: Optional[str] = None
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to save checkpoints and logs"},
    )

@dataclass
class PretrainConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self):
        if not self.dataset.train_path:
            raise ValueError("train_path must be specified in the dataset configuration.")
        if self.model.encoder_layers <= 0:
            raise ValueError("encoder_layers must be a positive integer.")
        if self.training.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if self.training.lr <= 0:
            raise ValueError("Learning rate must be a positive value.")
        if self.training.total_steps <= 0:
            raise ValueError("total_steps must be a positive integer.")
        if self.training.keep_last_n_checkpoints <= 0:
            raise ValueError("keep_last_n_checkpoints must be a positive integer.")
        if self.training.patience < -1:
            raise ValueError("patience must be -1 or non-negative.")

cs = ConfigStore.instance()
cs.store(name="pretrain_config", node=PretrainConfig)