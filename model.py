import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchtune.modules import MultiHeadAttention, RotaryPositionalEmbeddings


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.w3(self.w1(x) * F.silu(self.w2(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_ratio=4, dropout=0.0, checkpointing=False):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)

        # Create projection layers for attention
        q_proj = nn.Linear(dim, dim, bias=False)
        k_proj = nn.Linear(dim, (dim // num_heads) * num_kv_heads, bias=False)
        v_proj = nn.Linear(dim, (dim // num_heads) * num_kv_heads, bias=False)
        out_proj = nn.Linear(dim, dim, bias=False)

        # Create rotary embeddings
        pos_embeddings = RotaryPositionalEmbeddings(dim // num_heads)

        # Initialize MultiHeadAttention with all required components
        self.attn = MultiHeadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=dim // num_heads,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            output_proj=out_proj,
            pos_embeddings=pos_embeddings,
            is_causal=True,
            attn_dropout=dropout
        )

        self.norm2 = nn.RMSNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))
        self.dropout = nn.Dropout(dropout)
        self.checkpointing = checkpointing

    def forward(self, x, attn_mask=None, kv_cache=None, training_mode=False):
        def fn(x, attn_mask, kv_cache):  # allows use with checkpoint
            # Use kv_cache as key/value input if available, otherwise use input
            kv_input = kv_cache if kv_cache is not None else self.norm1(x)
            attn_out = self.attn(
                self.norm1(x), 
                y=kv_input,
                mask=attn_mask
            )
            x = x + self.dropout(attn_out)
            x = x + self.dropout(self.mlp(self.norm2(x)))
            return x, kv_input  # Return the kv_input as the new cache

        if self.checkpointing and self.training:
            return checkpoint(fn, x, attn_mask, kv_cache, use_reentrant=False)
        
        # During training, we still use fn but pass None as kv_cache
        # This ensures we use the same code path and logic
        return fn(x, attn_mask, None if training_mode else kv_cache)


class Transformer(nn.Module):
    def __init__(self, vocab_size, dim, depth, num_heads, num_kv_heads, mlp_ratio=4, dropout=0.0, checkpointing=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, num_kv_heads, mlp_ratio, dropout, checkpointing) for _ in range(depth)
        ])
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, attn_mask=None, kv_caches=None, training_mode=False):
        x = self.embed(input_ids)
        new_kv_caches = []
        
        for i, layer in enumerate(self.layers):
            kv_cache = None if kv_caches is None else kv_caches[i]
            x, new_kv = layer(x, attn_mask, kv_cache, training_mode)
            if not training_mode:
                new_kv_caches.append(new_kv)
            
        return self.head(self.norm(x)), new_kv_caches if not training_mode else None
