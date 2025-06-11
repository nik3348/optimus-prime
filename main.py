import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint

from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm

from pathlib import Path
from datasets import load_dataset
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


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


def rotary_emb(q, k, freqs):
    cos, sin = freqs
    if k.size(1) == 1:
        cos = cos.expand(-1, q.size(1), -1, -1)
        sin = sin.expand(-1, q.size(1), -1, -1)

    # Apply rotary embeddings to queries
    q_rot = (q * cos) + (rotate_half(q) * sin)

    # For keys, we need to handle the different number of heads
    # First apply to the original k tensor
    k_rot = (k * cos[..., :k.size(2), :]) + \
        (rotate_half(k) * sin[..., :k.size(2), :])

    return q_rot, k_rot


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._update_cos_sin_cache(max_seq_len)

    def _update_cos_sin_cache(self, seq_len):
        self.max_seq_len = max(self.max_seq_len, seq_len)
        t = torch.arange(self.max_seq_len,
                         device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x):
        seq_len = x.size(-2)
        if seq_len > self.max_seq_len:
            self._update_cos_sin_cache(seq_len)
        return self.cos_cached[..., :seq_len, :], self.sin_cached[..., :seq_len, :]


class GroupedQueryAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.dropout = dropout

        # Projections for Q, K, V
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, self.head_dim * num_kv_heads)
        self.v_proj = nn.Linear(dim, self.head_dim * num_kv_heads)
        self.o_proj = nn.Linear(dim, dim)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x, attn_mask=None, kv_cache=None):
        B, T, C = x.shape
        H = self.num_heads
        KV = self.num_kv_heads
        D = self.head_dim

        # Project Q, K, V
        q = self.q_proj(x).view(B, T, H, D)
        k = self.k_proj(x).view(B, T, KV, D)
        v = self.v_proj(x).view(B, T, KV, D)

        # Handle KV caching
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
            T = k.size(1)

        # Apply rotary embeddings
        freqs = self.rope(q)
        q, k = rotary_emb(q, k, freqs)

        # Repeat KV heads to match query heads
        k = k.repeat_interleave(self.num_queries_per_kv, dim=2)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=2)

        # Create causal mask if not provided
        if attn_mask is None:
            attn_mask = torch.tril(torch.ones((T, T), device=x.device))
            attn_mask = attn_mask.view(1, 1, T, T).expand(B, H, -1, -1)

        # Reshape tensors for attention
        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)  # [B, H, T, D]
        v = v.transpose(1, 2)  # [B, H, T, D]

        # Use PyTorch's built-in scaled dot product attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale
        )

        # Reshape back
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.o_proj(out), (k, v) if kv_cache is not None else None


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.linear3(F.silu(self.linear1(x)) * self.linear2(x))


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calculate RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Normalize and scale
        return x * rms * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, hidden_dim, dropout=0.1, use_checkpoint=True):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = GroupedQueryAttention(
            dim, num_heads, num_kv_heads, dropout)
        self.ln2 = RMSNorm(dim)
        self.ff = SwiGLU(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_checkpoint = use_checkpoint

    def forward(self, x, attn_mask=None, kv_cache=None):
        if self.use_checkpoint and self.training:
            attn_out, kv_cache = checkpoint(
                self.attn,
                self.ln1(x),
                attn_mask,
                kv_cache,
                use_reentrant=False
            )
            ffn_out = checkpoint(
                self.ff,
                self.ln2(x),
                use_reentrant=False
            )
        else:
            attn_out, kv_cache = self.attn(self.ln1(x), attn_mask, kv_cache)
            ffn_out = self.ff(self.ln2(x))

        x = x + self.dropout(attn_out + ffn_out)  # Parallel residual
        return x, kv_cache


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.pos_emb = nn.Parameter(torch.zeros(
            1, config.max_seq_len, config.dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(config.dim, config.num_heads, config.num_kv_heads,
                             config.hidden_dim, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.ln_f = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, mean=0.0,
                        std=0.02 * math.sqrt(self.config.dim))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02 *
                        math.sqrt(self.config.dim))
        nn.init.normal_(self.head.weight, mean=0.0,
                        std=0.02 * math.sqrt(self.config.dim))

    def forward(self, idx, kv_cache=None):
        B, T = idx.shape
        assert T <= self.config.max_seq_len, "Sequence too long"

        tok_emb = self.embed(idx)
        x = tok_emb + self.pos_emb[:, :T, :]

        attn_mask = torch.tril(torch.ones((T, T), device=idx.device))
        attn_mask = attn_mask.view(1, 1, T, T).expand(B, -1, -1, -1)

        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)

        new_kv_cache = []
        for block, block_kv_cache in zip(self.blocks, kv_cache):
            x, new_block_kv_cache = block(x, attn_mask, block_kv_cache)
            new_kv_cache.append(new_block_kv_cache)

        logits = self.head(self.ln_f(x))
        return logits, new_kv_cache


def prepare_batch(batch, tokenizer, max_length=128):
    # Process all texts at once for better efficiency
    texts = batch['text']

    # Calculate optimal batch size based on sequence lengths
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding='longest',
        return_tensors='pt',
        return_length=True
    )

    # Sort by length for more efficient padding
    lengths = encodings['length']
    sorted_indices = torch.argsort(lengths, descending=True)

    return {
        'input_ids': encodings['input_ids'][sorted_indices],
        'attention_mask': encodings['attention_mask'][sorted_indices] if 'attention_mask' in encodings else None
    }


def get_optimal_batch_size(model, config, device):
    """Dynamically determine optimal batch size based on available memory"""
    if not torch.cuda.is_available():
        return config.batch_size

    # Start with a small batch size
    batch_size = 1
    max_batch_size = config.batch_size

    while batch_size <= max_batch_size:
        try:
            # Create dummy input
            dummy_input = torch.randint(
                0, config.vocab_size, (batch_size, config.max_seq_len), device=device)

            # Try forward pass
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=config.use_amp):
                with torch.no_grad():
                    model(dummy_input)

            # If successful, try next batch size
            batch_size *= 2
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                # If OOM, return previous batch size
                torch.cuda.empty_cache()
                return batch_size // 2
            raise e

    return batch_size // 2


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_model(model, train_loader, val_loader, config: ModelConfig, device='cuda'):
    # Initialize optimizer with weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps
    )

    # Calculate total steps ensuring it's at least 1
    total_steps = max(1, len(train_loader) * config.num_epochs //
                      config.gradient_accumulation_steps)
    # Ensure warmup_steps is less than total_steps
    warmup_steps = min(config.warmup_steps, total_steps - 1)

    # Ensure we have at least one step for the main training phase
    if warmup_steps >= total_steps:
        warmup_steps = max(0, total_steps - 1)

    logger.info(
        f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps if total_steps > 0 else 0.0,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0
    )

    scaler = GradScaler(enabled=config.use_amp)
    best_val_loss = float('inf')
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_loss = AverageMeter()
        progress_bar = tqdm(
            train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs}')

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(progress_bar):
            try:
                input_ids = batch['input_ids'].to(device)

                with autocast(device_type='cuda', dtype=torch.float16, enabled=config.use_amp):
                    logits, _ = model(input_ids)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=0
                    )
                    loss = loss / config.gradient_accumulation_steps

                if torch.isnan(loss):
                    logger.error("NaN loss detected! Skipping batch.")
                    continue

                scaler.scale(loss).backward()

                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.gradient_clip_val)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                train_loss.update(
                    loss.item() * config.gradient_accumulation_steps)
                progress_bar.set_postfix({
                    'loss': f'{train_loss.avg:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })

            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error("GPU OOM in batch. Skipping batch.")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    continue
                raise e

        # Validation
        model.eval()
        val_loss = AverageMeter()
        with torch.no_grad():
            for batch in val_loader:
                try:
                    input_ids = batch['input_ids'].to(device)

                    with autocast(device_type='cuda', dtype=torch.float16, enabled=config.use_amp):
                        logits, _ = model(input_ids)
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = input_ids[..., 1:].contiguous()
                        loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=0
                        )
                    val_loss.update(loss.item())
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.error(
                            "GPU OOM in validation batch. Skipping batch.")
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        continue
                    raise e

        logger.info(
            f'Epoch {epoch + 1}: Train Loss: {train_loss.avg:.4f}, Val Loss: {val_loss.avg:.4f}')

        if val_loss.avg < best_val_loss:
            best_val_loss = val_loss.avg
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': train_loss.avg,
                'val_loss': val_loss.avg,
            }, save_dir / 'best_model.pt')
            logger.info(
                f'Saved best model with validation loss: {val_loss.avg:.4f}')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Initialize model and enable optimizations
    config = ModelConfig()
    model = DecoderOnlyTransformer(config).to(device)

    # Load model weights if they exist
    if (Path(config.save_dir) / 'best_model.pt').exists():
        checkpoint = torch.load(Path(config.save_dir) / 'best_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from epoch {checkpoint['epoch']}")
    else:
        logger.info("No existing model found. Starting from scratch.")

    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2',
                                              model_max_length=config.max_seq_len,
                                              padding_side='right')
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset with streaming
    fw = load_dataset("HuggingFaceFW/fineweb-edu",
                      name="CC-MAIN-2024-10",
                      split="train",
                      streaming=True)

    # Prepare data with better memory management
    train_data = list(fw.take(1000))
    val_data = list(fw.skip(1000).take(1000))

    # Get optimal batch size
    optimal_batch_size = get_optimal_batch_size(model, config, device)
    logger.info(f"Using optimal batch size: {optimal_batch_size}")
    config.batch_size = optimal_batch_size

    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=lambda x: prepare_batch(
            {'text': [item['text'] for item in x]}, tokenizer)
    )

    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=lambda x: prepare_batch(
            {'text': [item['text'] for item in x]}, tokenizer)
    )

    train_model(model, train_loader, val_loader, config, device)


if __name__ == "__main__":
    main()
