# unimol_tools/ta/config.py

import warnings
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

@dataclass
class DatasetConfig:
    train_lmdb: str = MISSING
    valid_lmdb: Optional[str] = None
    dict_path: Optional[str] = None
    remove_hydrogen: bool = False
    max_atoms: int = 256
    noise_type: str = "trunc_normal"
    noise: float = 0.1
    mask_prob: float = 0.15
    leave_unmasked_prob: float = 0.1
    random_token_prob: float = 0.1

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
    weight_decay: float = 0.01
    epochs: int = 10
    save_every_n_epoch: int = 1
    use_amp: bool = True
    log_wandb: bool = True
    wandb_project: str = "unimol_pretrain"
    local_rank: int = 0
    seed: int = 42

@dataclass
class PretrainConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self):
        if self.dataset.train_lmdb is MISSING:
            raise ValueError("train_lmdb must be specified in the dataset configuration.")
        if self.model.encoder_layers <= 0:
            raise ValueError("encoder_layers must be a positive integer.")
        if self.training.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if self.training.lr <= 0:
            raise ValueError("Learning rate must be a positive value.")

cs = ConfigStore.instance()
cs.store(name="pretrain_config", node=PretrainConfig)