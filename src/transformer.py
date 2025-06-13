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

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.embedding_dim)
        self.norm2 = nn.RMSNorm(config.embedding_dim)
        kv_compression_dim = int(
            config.embedding_dim * config.kv_compression_ratio)
        q_compression_dim = int(
            config.embedding_dim * config.q_compression_ratio
        )
        self.attn = MLA(
            embedding_dim=config.embedding_dim,
            num_attention_heads=config.num_attention_heads,
            kv_compression_dim=kv_compression_dim,
            q_compression_dim=q_compression_dim,
            rope_seq_len=config.rope_seq_len
        )
        self.mlp = SwiGLU(
            config.embedding_dim,
            config.embedding_dim * config.mlp_ratio
        )

    def forward(self, x, mask=None, past_kv_latent=None, return_kv_cache=False, return_raw_kv_cache=False):
        """
        Forward pass for the MLA-based transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, dim).
            mask (torch.Tensor, optional): Attention mask.
            past_kv_latent (torch.Tensor, optional): Cached latent for KV.
            return_kv_cache (bool): Whether to return KV cache.
            return_raw_kv_cache (bool): Whether to return the raw compressed kv_latent for caching.

        Returns:
            torch.Tensor: Output tensor.
            (optionally) tuple: Output and KV cache if return_kv_cache is True.
        """
        normed_x = self.norm1(x)
        attn_out, kv_latent = self.attn(
            normed_x, mask=mask, past_kv_latent=past_kv_latent, return_kv_cache=True)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        if return_kv_cache:
            # Always return the raw kv_latent (compressed, [B, 1, kv_compression_dim])
            return x, kv_latent[:, -1:, :]
        else:
            return x


class TransformerBlock(nn.Module):
    """
    Standard transformer block with MultiHeadAttention and SwiGLU feedforward.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        head_dim = config.embedding_dim // config.num_attention_heads

        # Create projection layers for attention
        q_proj = nn.Linear(
            config.embedding_dim, config.embedding_dim, bias=False)
        k_proj = nn.Linear(
            config.embedding_dim, head_dim * config.num_kv_heads, bias=False)
        v_proj = nn.Linear(
            config.embedding_dim, head_dim * config.num_kv_heads, bias=False)
        out_proj = nn.Linear(
            config.embedding_dim, config.embedding_dim, bias=False)

        # Create rotary embeddings
        pos_embeddings = RotaryPositionalEmbeddings(head_dim)

        # Initialize MultiHeadAttention with all required components
        self.attn = MultiHeadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.num_attention_heads,
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

        self.norm1 = nn.RMSNorm(config.embedding_dim)
        self.norm2 = nn.RMSNorm(config.embedding_dim)
        self.mlp = SwiGLU(
            config.embedding_dim,
            config.embedding_dim * config.mlp_ratio
        )
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
