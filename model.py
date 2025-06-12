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
        head_dim = config.dim // config.num_heads

        # Create projection layers for attention
        q_proj = nn.Linear(config.dim, config.dim, bias=False)
        k_proj = nn.Linear(
            config.dim, head_dim * config.num_kv_heads, bias=False)
        v_proj = nn.Linear(
            config.dim, head_dim * config.num_kv_heads, bias=False)
        out_proj = nn.Linear(config.dim, config.dim, bias=False)

        # Create rotary embeddings
        pos_embeddings = RotaryPositionalEmbeddings(head_dim)

        # Initialize MultiHeadAttention with all required components
        self.attn = MultiHeadAttention(
            embed_dim=config.dim,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=head_dim,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            output_proj=out_proj,
            pos_embeddings=pos_embeddings,
            q_norm=nn.RMSNorm(head_dim),
            k_norm=nn.RMSNorm(head_dim),
            is_causal=True,
        )

        self.norm1 = nn.RMSNorm(config.dim)
        self.norm2 = nn.RMSNorm(config.dim)
        self.mlp = SwiGLU(config.dim, int(config.dim * mlp_ratio))
        self.checkpointing = config.checkpointing

    def forward(self, x):
        normed_x = self.norm1(x)

        if self.checkpointing and self.training:
            attn_out = checkpoint(
                self.attn, normed_x, normed_x, use_reentrant=False
            )
        else:
            attn_out = self.attn(normed_x, y=normed_x)

        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


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

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))
