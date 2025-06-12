from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    vocab_size: int = 50257  # GPT-2 tokenizer vocabulary size
    dim: int = 1024
    depth: int = 8
    num_heads: int = 8
    num_kv_heads: int = 2
    checkpointing: bool = True

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Training parameters
    batch_size: int = 8
    max_length: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 5
    train_val_split: float = 0.9
    gradient_clip_val: float = 1.0

    # Dataset parameters
    dataset_name: str = "databricks/databricks-dolly-15k"
    dataset_size: int = 0
    num_workers: int = 4

    # Paths
    checkpoint_dir: str = "checkpoints"
    tensorboard_dir: str = "runs"
