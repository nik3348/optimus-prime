import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchtune.modules import MultiHeadAttention, RotaryPositionalEmbeddings
from config import ModelConfig


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.w3(self.w1(x) * F.silu(self.w2(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, mlp_ratio=4):
        super().__init__()
        # Create projection layers for attention
        q_proj = nn.Linear(config.dim, config.dim, bias=False)
        k_proj = nn.Linear(
            config.dim, (config.dim // config.num_heads) * config.num_kv_heads, bias=False)
        v_proj = nn.Linear(
            config.dim, (config.dim // config.num_heads) * config.num_kv_heads, bias=False)
        out_proj = nn.Linear(config.dim, config.dim, bias=False)

        # Create rotary embeddings
        pos_embeddings = RotaryPositionalEmbeddings(
            config.dim // config.num_heads
        )

        # Initialize MultiHeadAttention with all required components
        self.attn = MultiHeadAttention(
            embed_dim=config.dim,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.dim // config.num_heads,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            output_proj=out_proj,
            pos_embeddings=pos_embeddings,
            is_causal=True,
        )

        self.norm1 = nn.RMSNorm(config.dim)
        self.norm2 = nn.RMSNorm(config.dim)
        self.mlp = SwiGLU(config.dim, int(config.dim * mlp_ratio))
        self.checkpointing = config.checkpointing

    def _forward_inner(self, x, kv_input=None):
        """
        Common forward pass logic for both training (with/without checkpointing) and inference.
        """
        normed_x = self.norm1(x)

        # Determine kv_input
        actual_kv_input = normed_x if kv_input is None else kv_input
        attn_out = self.attn(normed_x, y=actual_kv_input)

        x = x + attn_out
        x = x + self.mlp(self.norm2(x))

        return x, actual_kv_input  # Return kv_input to pass it up for actual caching

    def forward(self, x, kv_cache=None):
        if self.checkpointing and self.training:
            # When checkpointing, we pass None for kv_cache, meaning kv_input will be normed_x
            # and attn_mask is passed directly.
            x_out, _ = checkpoint(self._forward_inner, x, use_reentrant=False)
            # Return None for kv_cache as it's not cached during checkpointed training
            return x_out, None

        # For regular training (no checkpointing) or inference
        x_out, updated_kv_input = self._forward_inner(x, kv_cache)
        return x_out, updated_kv_input


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig, mlp_ratio=4):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([
            TransformerBlock(config, mlp_ratio) for _ in range(config.depth)
        ])
        self.norm = nn.RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, kv_caches=None):
        x = self.embed(input_ids)
        new_kv_caches = []

        for i, layer in enumerate(self.layers):
            kv_cache = None if kv_caches is None else kv_caches[i]
            x, new_kv = layer(x, kv_cache)
            if not self.training:  # Only cache during inference
                new_kv_caches.append(new_kv)

        return self.head(self.norm(x)), new_kv_caches if not self.training else None
