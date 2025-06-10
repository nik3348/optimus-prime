import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Rotary Embeddings (2D rotation of query/key)
def rotary_emb(q, k, freqs):
    cos, sin = freqs
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i , j -> i j", t, inv_freq)
        self.register_buffer("cos", freqs.cos()[None, None, :, :])
        self.register_buffer("sin", freqs.sin()[None, None, :, :])

    def forward(self, x):
        return self.cos[..., :x.size(-2), :], self.sin[..., :x.size(-2), :]


class MultiQueryAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, head_dim)
        self.v_proj = nn.Linear(dim, head_dim)
        self.o_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(head_dim)

    def forward(self, x, attn_mask=None):
        B, T, C = x.shape
        H = self.num_heads
        D = C // H

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)   # (B, H, T, D)
        k = self.k_proj(x).unsqueeze(1)                       # (B, 1, T, D)
        v = self.v_proj(x).unsqueeze(1)                       # (B, 1, T, D)

        freqs = self.rope(q)
        q, k = rotary_emb(q, k, freqs)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        attn = attn.masked_fill(attn_mask == 0, float('-inf')) if attn_mask is not None else attn
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, C)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.linear3(F.silu(self.linear1(x)) * self.linear2(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiQueryAttention(dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = SwiGLU(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.dropout(self.attn(self.ln1(x), attn_mask))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, dim, num_heads, num_layers, hidden_dim, dropout=0.1, max_seq_len=2048):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.max_seq_len = max_seq_len

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.max_seq_len, "Sequence too long"

        tok_emb = self.embed(idx)
        x = tok_emb + self.pos_emb[:, :T, :]

        # Causal mask
        attn_mask = torch.tril(torch.ones((T, T), device=idx.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        for block in self.blocks:
            x = block(x, attn_mask)

        logits = self.head(self.ln_f(x))
        return logits

# Example Usage
model = DecoderOnlyTransformer(
    vocab_size=50257,  # GPT-2 vocab size
    dim=768,
    num_heads=12,
    num_layers=12,
    hidden_dim=768 * 4,
    dropout=0.1
)

x = torch.randint(0, 50257, (2, 128))  # (batch_size, seq_len)
logits = model(x)
print(logits.shape)  # Expected: (2, 128, 50257)
