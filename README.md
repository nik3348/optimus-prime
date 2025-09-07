# Optimus Prime

A modern, efficient transformer-based language model with several optimizations and best practices. The model features Multi-head latent attention (MLA), rotary embeddings, and other architectural improvements for better performance and efficiency.

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

- **Multi-head latent attention (MLA)**: A novel attention mechanism that uses low-rank projections and separate RoPE embeddings for queries and keys to improve performance.
- **Rotary Position Embeddings (RoPE)**: Better position encoding that improves model performance on longer sequences.
- **SwiGLU Activation**: Modern activation function that improves model performance.
- **RMSNorm**: Efficient normalization layer.
- **Gradient Checkpointing**: Memory-efficient training through gradient checkpointing.
- **Mixed Precision Training**: Automatic Mixed Precision (AMP) support for faster training.
- **Efficient KV Caching**: Optimized inference with KV and KR caching for the MLA module.
- **Top-P (Nucleus) Sampling**: High-quality text generation with nucleus sampling.

## Model Architecture

The model is based on a decoder-only transformer architecture with the following components:

- **MLA Transformer Block**: Each block contains:
    - **Multi-head latent attention (MLA)**: The core attention module.
    - **SwiGLU Feed-Forward Network**: For non-linear transformations.
    - **RMSNorm**: Applied before the attention and MLP layers.
    - **Dropout**: For regularization.
- The model can be configured to use either the `MLATransformerBlock` or a standard `TransformerBlock`.

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
    vocab_size: int = 50257
    embedding_dim: int = 768
    num_layers: int = 8
    num_attention_heads: int = 12
    num_kv_heads: int = 2
    checkpointing: bool = True
    compression_ratio: float = 0.5
    rope_seq_len: int = 2048
    mlp_ratio: float = 4
    dropout: float = 0.15

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Training parameters
    batch_size: int = 64
    max_length: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    num_epochs: int = 3
    train_val_split: float = 0.9
    gradient_clip_val: float = 1.0

    # Dataset parameters
    dataset_name: str = "wikimedia/wikipedia"
    dataset_size: int = 200000
    num_workers: int = 4

    # Paths
    checkpoint_dir: str = "checkpoints"
    tensorboard_dir: str = "runs"
```

## Training Data

The model is trained on the Wikipedia dataset (`wikimedia/wikipedia`), with configurable dataset size and train/validation split. The training script includes efficient data loading and preprocessing with multi-worker support.

## Performance Optimizations

### Memory Efficiency

- Gradient checkpointing
- Dynamic batch size optimization
- Efficient KV and KR caching during inference in the MLA module.

### Training Speed

- Automatic Mixed Precision (AMP)
- Optimized attention implementation with MLA.
- Efficient data loading with prefetching.

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
├── src/               # Source code for model components
│   ├── attention.py   # MLA implementation
│   ├── feedforward.py # SwiGLU implementation
│   └── transformer.py # Transformer block implementations
├── checkpoints/       # Saved model checkpoints
├── runs/              # Training logs and metrics
├── pyproject.toml     # Poetry configuration and dependencies
└── poetry.lock        # Locked dependencies
```
