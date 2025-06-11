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
    def __init__(self, config: ModelConfig, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.dim)

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
            attn_dropout=dropout
        )

        self.norm2 = nn.RMSNorm(config.dim)
        self.mlp = SwiGLU(config.dim, int(config.dim * mlp_ratio))
        self.dropout = nn.Dropout(dropout)
        self.checkpointing = config.checkpointing

    def _forward_no_cache(self, x, attn_mask):
        """Forward pass without caching, used for checkpointing during training."""
        normed_x = self.norm1(x)
        attn_out = self.attn(normed_x, y=normed_x, mask=attn_mask)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

    def forward(self, x, attn_mask=None, kv_cache=None):
        if self.checkpointing and self.training:
            x = checkpoint(self._forward_no_cache, x,
                           attn_mask, use_reentrant=False)
            return x, None

        # Normalize input once and reuse
        normed_x = self.norm1(x)
        kv_input = kv_cache if kv_cache is not None else normed_x

        # Attention
        attn_out = self.attn(normed_x, y=kv_input, mask=attn_mask)
        x = x + self.dropout(attn_out)

        # MLP
        x = x + self.dropout(self.mlp(self.norm2(x)))

        return x, kv_input


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([
            TransformerBlock(config, mlp_ratio, dropout) for _ in range(config.depth)
        ])
        self.norm = nn.RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, attn_mask=None, kv_caches=None):
        x = self.embed(input_ids)
        new_kv_caches = []

        for i, layer in enumerate(self.layers):
            kv_cache = None if kv_caches is None else kv_caches[i]
            x, new_kv = layer(x, attn_mask, kv_cache)
            if not self.training:  # Only cache during inference
                new_kv_caches.append(new_kv)

        return self.head(self.norm(x)), new_kv_caches if not self.training else None
