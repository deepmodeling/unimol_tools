from dataclasses import dataclass, field
from typing import Optional, Tuple

from hydra.core.config_store import ConfigStore


@dataclass
class DatasetConfig:
    train_path: str = field(
        default="",
        metadata={"help": "Path to the training dataset."},
    )
    valid_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the validation dataset."},
    )
    data_type: str = field(
        default="lmdb",
        metadata={"help": "Dataset format, e.g. 'lmdb', 'csv', 'txt', 'smi', or 'sdf'."},
    )
    smiles_column: str = field(
        default="smi",
        metadata={"help": "Column name for SMILES when reading CSV data."},
    )
    dict_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to the dictionary file."},
    )
    remove_hydrogen: bool = field(
        default=False,
        metadata={"help": "Remove hydrogen atoms from molecules."},
    )
    max_atoms: int = field(
        default=256,
        metadata={"help": "Maximum number of atoms per molecule."},
    )
    noise_type: str = field(
        default="uniform",
        metadata={"help": "Type of noise added to coordinates during masking."},
    )
    noise: float = field(
        default=1.0,
        metadata={"help": "Magnitude of coordinate noise."},
    )
    mask_prob: float = field(
        default=0.15,
        metadata={"help": "Probability of masking an atom token."},
    )
    leave_unmasked_prob: float = field(
        default=0.05,
        metadata={"help": "Probability of keeping a masked token unchanged."},
    )
    random_token_prob: float = field(
        default=0.05,
        metadata={"help": "Probability of replacing a masked token with a random token."},
    )
    add_2d: bool = field(
        default=True,
        metadata={"help": "Append a 2D conformer generated from the SMILES string at runtime."},
    )
    num_conformers: int = field(
        default=10,
        metadata={"help": "Number of 3D conformers to generate per molecule during preprocessing."},
    )
    preprocess_workers: int = field(
        default=8,
        metadata={
            "help": "Number of worker processes to use when converting datasets to LMDB."
        },
    )
    stats_workers: int = field(
        default=8,
        metadata={
            "help": "Number of CPU workers to scan LMDBs for building dictionaries and distance statistics."
        },
    )

@dataclass
class ModelConfig:
    model_name: str = field(
        default="UniMol",
        metadata={"help": "Model architecture to use."},
    )
    encoder_layers: int = field(
        default=15,
        metadata={"help": "Number of transformer encoder layers."},
    )
    encoder_embed_dim: int = field(
        default=512,
        metadata={"help": "Embedding dimension of the encoder."},
    )
    encoder_ffn_embed_dim: int = field(
        default=2048,
        metadata={"help": "Hidden dimension of the encoder feed-forward network."},
    )
    encoder_attention_heads: int = field(
        default=64,
        metadata={"help": "Number of attention heads in the encoder."},
    )
    dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout rate applied after the feed-forward network."},
    )
    emb_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout rate applied to embeddings."},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout rate for attention weights."},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout after activation in the feed-forward network."},
    )
    pooler_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout applied in the pooling layer."},
    )
    max_seq_len: int = field(
        default=512,
        metadata={"help": "Maximum input sequence length."},
    )
    activation_fn: str = field(
        default="gelu",
        metadata={"help": "Activation function for the feed-forward network."},
    )
    pooler_activation_fn: str = field(
        default="tanh",
        metadata={"help": "Activation function for the pooling layer."},
    )
    post_ln: bool = field(
        default=False,
        metadata={"help": "Use Post-LayerNorm instead of Pre-LayerNorm."},
    )
    masked_token_loss: float = field(
        default=1.0,
        metadata={"help": "Weight of the masked token prediction loss."},
    )
    masked_coord_loss: float = field(
        default=5.0,
        metadata={"help": "Weight of the masked coordinate prediction loss."},
    )
    masked_dist_loss: float = field(
        default=10.0,
        metadata={"help": "Weight of the masked distance prediction loss."},
    )
    x_norm_loss: float = field(
        default=0.01,
        metadata={"help": "Weight of the atom-wise representation norm regularization."},
    )
    delta_pair_repr_norm_loss: float = field(
        default=0.01,
        metadata={"help": "Weight of the pair representation difference norm regularization."},
    )

@dataclass
class TrainingConfig:
    batch_size: int = field(
        default=16,
        metadata={"help": "Per-GPU batch size; effective batch size is n_gpu * batch_size * update_freq."},
    )
    update_freq: int = field(
        default=1,
        metadata={"help": "Number of steps to accumulate gradients before updating (update frequency)."},
    )
    lr: float = field(
        default=1e-4,
        metadata={"help": "Initial learning rate."},
    )
    weight_decay: float = field(
        default=1e-4,
        metadata={"help": "Weight decay for Adam optimizer."},
    )
    adam_betas: Tuple[float, float] = field(
        default=(0.9, 0.99),
        metadata={"help": "Beta coefficients for Adam optimizer."},
    )
    adam_eps: float = field(
        default=1e-6,
        metadata={"help": "Epsilon for Adam optimizer."},
    )
    clip_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Maximum gradient norm for clipping."},
    )
    epochs: int = field(
        default=0,
        metadata={"help": "Number of epochs; 0 to disable and rely on total_steps."},
    )
    total_steps: int = field(
        default=1000000,
        metadata={"help": "Maximum number of optimizer update steps."},
    )
    warmup_steps: int = field(
        default=10000,
        metadata={"help": "Number of steps to linearly warm up the learning rate."},
    )
    log_every_n_steps: int = field(
        default=10,
        metadata={"help": "Log training metrics every N update steps."},
    )
    save_every_n_steps: int = field(
        default=10000,
        metadata={"help": "Save a checkpoint every N update steps."},
    )
    keep_last_n_checkpoints: int = field(
        default=5,
        metadata={"help": "How many step checkpoints to keep."},
    )
    patience: int = field(
        default=-1,
        metadata={"help": "Early stop patience based on validation loss; -1 disables."},
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "Use fp16 mixed precision training."},
    )
    fp16_init_scale: float = field(
        default=4,
        metadata={"help": "Initial loss scale for fp16 training."},
    )
    fp16_scale_window: int = field(
        default=256,
        metadata={"help": "Steps without overflow before increasing the loss scale."},
    )
    num_workers: int = field(
        default=8,
        metadata={"help": "Number of worker processes for data loading."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."},
    )
    resume: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a checkpoint to resume training from."},
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to save checkpoints and logs."},
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
        if self.training.update_freq <= 0:
            raise ValueError("update_freq must be a positive integer.")
        if self.dataset.preprocess_workers <= 0:
            raise ValueError("preprocess_workers must be a positive integer.")
        if self.dataset.stats_workers <= 0:
            raise ValueError("stats_workers must be a positive integer.")
        if self.training.lr <= 0:
            raise ValueError("Learning rate must be a positive value.")
        if self.training.total_steps <= 0:
            raise ValueError("total_steps must be a positive integer.")
        if self.training.keep_last_n_checkpoints <= 0:
            raise ValueError("keep_last_n_checkpoints must be a positive integer.")
        if self.training.patience < -1:
            raise ValueError("patience must be -1 or non-negative.")
        if self.training.fp16_init_scale <= 0:
            raise ValueError("fp16_init_scale must be a positive value.")
        if self.training.fp16_scale_window <= 0:
            raise ValueError("fp16_scale_window must be a positive integer.")

cs = ConfigStore.instance()
cs.store(name="pretrain_config", node=PretrainConfig)