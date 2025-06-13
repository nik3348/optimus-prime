import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchtune.modules import MultiHeadAttention, RotaryPositionalEmbeddings

from .attention import MLA
from .feedforward import SwiGLU
from config import ModelConfig

class MLATransformerBlock(nn.Module):
    """
    Transformer block leveraging MLA attention and SwiGLU feedforward modules.
    """
    def __init__(self, config: ModelConfig, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.dim)
        self.norm2 = nn.RMSNorm(config.dim)
        self.attn = MLA(
            dim=config.dim,
            num_heads=config.num_heads,
            kv_compression_dim=config.dim // 2,  # example compression, adjust as needed
            q_compression_dim=config.dim // 2,   # example compression, adjust as needed
            rope_seq_len=2048
        )
        self.mlp = SwiGLU(config.dim, int(config.dim * mlp_ratio))

    def forward(self, x, mask=None, past_kv_latent=None, return_kv_cache=False):
        """
        Forward pass for the MLA-based transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, dim).
            mask (torch.Tensor, optional): Attention mask.
            past_kv_latent (torch.Tensor, optional): Cached latent for KV.
            return_kv_cache (bool): Whether to return KV cache.

        Returns:
            torch.Tensor: Output tensor.
            (optionally) tuple: Output and KV cache if return_kv_cache is True.
        """
        normed_x = self.norm1(x)
        attn_out = self.attn(
            normed_x, mask=mask, past_kv_latent=past_kv_latent, return_kv_cache=return_kv_cache)
        if return_kv_cache:
            attn_out, new_kv_latent = attn_out[0], attn_out[1]
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        if return_kv_cache:
            return x, new_kv_latent
        else:
            return x


class TransformerBlock(nn.Module):
    """
    Standard transformer block with MultiHeadAttention and SwiGLU feedforward.
    """
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
        """
        Forward pass for the standard transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, dim).

        Returns:
            torch.Tensor: Output tensor after attention and feedforward.
        """
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
