from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    vocab_size: int = 50257
    embedding_dim: int = 512
    num_layers: int = 6
    num_attention_heads: int = 8
    num_kv_heads: int = 2
    checkpointing: bool = True
    kv_compression_ratio: float = 0.5
    q_compression_ratio: float = 0.5
    rope_seq_len: int = 2048
    mlp_ratio: float = 4


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Training parameters
    batch_size: int = 8
    max_length: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    num_epochs: int = 5
    train_val_split: float = 0.9
    gradient_clip_val: float = 1.0

    # Dataset parameters
    dataset_name: str = "databricks/databricks-dolly-15k"
    dataset_size: int = 0
    num_workers: int = 4

    # Paths
    checkpoint_dir: str = "data/checkpoints"
    tensorboard_dir: str = "data/runs"
