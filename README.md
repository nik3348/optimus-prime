# Optimus Prime

A modern, efficient transformer-based language model with several optimizations and best practices. The model features grouped query attention, rotary embeddings, and other architectural improvements for better performance and efficiency.

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

## Requirements

This project uses Poetry for dependency management. Make sure you have Poetry installed on your system. If not, you can install it following the [official installation guide](https://python-poetry.org/docs/#installation).

### Installation

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
python main.py
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
python run_model.py
```

The inference script provides an interactive interface where you can:
- Enter prompts and get generated text
- Control generation parameters (temperature, top-p)
- Generate text with efficient KV caching

## Model Configuration

The model can be configured through the `ModelConfig` class in `main.py`. Key parameters include:

```python
@dataclass
class ModelConfig:
    vocab_size: int = 50257
    dim: int = 768
    num_heads: int = 12
    num_kv_heads: int = 4
    num_layers: int = 12
    hidden_dim: int = 3072
    dropout: float = 0.1
    max_seq_len: int = 2048
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 1
    save_dir: str = "checkpoints"
    use_amp: bool = True
    warmup_steps: int = 1000
    gradient_clip_val: float = 1.0
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
```

## Training Data

The model is trained on the FineWeb-Edu dataset, specifically using the CC-MAIN-2024-10 split. The training script includes efficient data loading and preprocessing.

## Performance Optimizations

1. **Memory Efficiency**:
   - Gradient checkpointing
   - Dynamic batch size optimization
   - Efficient KV caching during inference

2. **Training Speed**:
   - Automatic Mixed Precision (AMP)
   - Optimized attention implementation
   - Efficient data loading with prefetching

3. **Inference Quality**:
   - Top-p (nucleus) sampling
   - Temperature control
   - Proper handling of special tokens

## File Structure

- `main.py`: Contains the model architecture and training code
- `run_model.py`: Provides an interface for model inference
- `checkpoints/`: Directory for saved model checkpoints
- `pyproject.toml`: Poetry configuration and dependencies

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
