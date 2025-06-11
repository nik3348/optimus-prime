import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchtune.modules import RotaryPositionalEmbeddings


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.w3(self.w1(x) * F.silu(self.w2(x)))


class FlashMQAttention(nn.Module):
    def __init__(self, dim, num_heads, kv_heads=None, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kv_heads = kv_heads or 1
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout
        self.rotary = RotaryPositionalEmbeddings(self.head_dim)

    def forward(self, x, attn_mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads,
                                self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.kv_heads,
                                self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kv_heads,
                                self.head_dim).transpose(1, 2)

        k = k.repeat_interleave(self.num_heads // self.kv_heads, dim=1)
        v = v.repeat_interleave(self.num_heads // self.kv_heads, dim=1)

        # Apply rotary embeddings using torchtune's implementation
        q = self.rotary(q)
        k = self.rotary(k)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask, self.dropout if self.training else 0.0, is_causal=True)
        return self.out_proj(out.transpose(1, 2).reshape(B, T, C))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.0, checkpointing=False):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = FlashMQAttention(dim, num_heads)
        self.norm2 = nn.RMSNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))
        self.dropout = nn.Dropout(dropout)
        self.checkpointing = checkpointing

    def forward(self, x, attn_mask=None):
        def fn(x, attn_mask):  # allows use with checkpoint
            x = x + self.dropout(self.attn(self.norm1(x), attn_mask))
            x = x + self.dropout(self.mlp(self.norm2(x)))
            return x
        return checkpoint(fn, x, attn_mask, use_reentrant=False) if self.checkpointing and self.training else fn(x, attn_mask)


class Transformer(nn.Module):
    def __init__(self, vocab_size, dim, depth, num_heads, mlp_ratio=4, dropout=0.0, checkpointing=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_ratio, dropout, checkpointing) for _ in range(depth)
        ])
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, attn_mask=None):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, attn_mask)
        return self.head(self.norm(x))
