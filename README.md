# Optimus Prime

A modern, efficient transformer-based language model with several optimizations and best practices. The model features grouped query attention, rotary embeddings, and other architectural improvements for better performance and efficiency.

## Table of Contents

- [Features](#features)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Model Configuration](#model-configuration)
- [Training Data](#training-data)
- [Performance Optimizations](#performance-optimizations)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Grouped Query Attention (GQA)**: Reduces memory usage while maintaining model quality by sharing key and value heads across multiple query heads
- **Rotary Position Embeddings (RoPE)**: Better position encoding that improves model performance on longer sequences
- **SwiGLU Activation**: Modern activation function that improves model performance
- **RMSNorm**: Efficient normalization layer
- **Gradient Checkpointing**: Memory-efficient training through gradient checkpointing
- **Mixed Precision Training**: Automatic Mixed Precision (AMP) support for faster training
- **Efficient KV Caching**: Optimized inference with KV caching
- **Top-P (Nucleus) Sampling**: High-quality text generation with nucleus sampling

## Model Architecture

The model is based on a decoder-only transformer architecture with the following components:

- Multi-head attention with grouped query attention
- Rotary position embeddings
- SwiGLU feed-forward networks
- RMSNorm for normalization
- Configurable model size and hyperparameters

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- Poetry for dependency management

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/optimus-prime.git
cd optimus-prime
```

2. Install dependencies using Poetry:

```bash
poetry install
```

3. Activate the virtual environment:

```bash
poetry shell
```

## Usage

### Training

To train the model:

```bash
python train.py
```

The training script includes:

- Automatic mixed precision training
- Gradient checkpointing
- Dynamic batch size optimization
- Learning rate scheduling with warmup
- Model checkpointing
- Validation during training

### Inference

To run the model for text generation:

```bash
python run.py
```

The inference script provides an interactive interface where you can:

- Enter prompts and get generated text
- Control generation parameters (temperature, top-p)
- Generate text with efficient KV caching

## Model Configuration

The model can be configured through the `ModelConfig` and `TrainingConfig` classes in `config.py`. Key parameters include:

```python
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
    dataset_size: int = 1000
    num_workers: int = 4

    # Paths
    checkpoint_dir: str = "checkpoints"
    tensorboard_dir: str = "runs"
```

## Training Data

The model is trained on the Databricks Dolly 15k dataset, with configurable dataset size and train/validation split. The training script includes efficient data loading and preprocessing with multi-worker support.

## Performance Optimizations

### Memory Efficiency

- Gradient checkpointing
- Dynamic batch size optimization
- Efficient KV caching during inference

### Training Speed

- Automatic Mixed Precision (AMP)
- Optimized attention implementation
- Efficient data loading with prefetching

### Inference Quality

- Top-p (nucleus) sampling
- Temperature control
- Proper handling of special tokens

## Project Structure

```
optimus-prime/
├── train.py           # Training script
├── run.py             # Model inference interface
├── model.py           # Model architecture implementation
├── config.py          # Configuration and hyperparameters
├── checkpoints/       # Saved model checkpoints
├── runs/              # Training logs and metrics
├── pyproject.toml     # Poetry configuration and dependencies
└── poetry.lock        # Locked dependencies
```
