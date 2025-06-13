import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings


class MLA(nn.Module):
    """
    Multi-Latent Attention (MLA) module with compressed key, value, and query projections,
    rotary positional embeddings, and decoupled RoPE components for keys and queries.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        kv_compression_dim: int,
        q_compression_dim: int,
        rope_seq_len: int,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rope = RotaryPositionalEmbeddings(
            dim=self.head_dim, base=10000, max_seq_len=rope_seq_len)

        # Projections for compressed KV
        self.kv_down = nn.Linear(dim, kv_compression_dim, bias=False)  # W_D_KV
        self.kv_up_k = nn.Linear(
            kv_compression_dim, num_heads * self.head_dim, bias=False)  # W_U_K
        self.kv_up_v = nn.Linear(
            kv_compression_dim, num_heads * self.head_dim, bias=False)  # W_U_V

        # Projections for compressed Q
        self.q_down = nn.Linear(dim, q_compression_dim, bias=False)  # W_D_Q
        self.q_up = nn.Linear(q_compression_dim, num_heads *
                              self.head_dim, bias=False)  # W_U_Q

        # Linear for the decoupled RoPE component
        self.k_rope_proj = nn.Linear(dim, self.head_dim, bias=False)   # W_K_R
        self.q_rope_proj = nn.Linear(dim, self.head_dim, bias=False)   # W_Q_R

        # Output projection
        self.out_proj = nn.Linear(
            num_heads * self.head_dim, dim, bias=False)  # W_O

    def forward(self, x, mask=None, past_kv_latent=None, return_kv_cache=False):
        """
        Forward pass for the MLA attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, dim).
            mask (torch.Tensor, optional): Attention mask of shape (batch, seq_len) or None.
            past_kv_latent (torch.Tensor, optional): Cached tensor from previous steps, shape (batch, T_past, kv_compression_dim).
            return_kv_cache (bool): If True, returns updated key/value cache.

        Returns:
            torch.Tensor: Output tensor after attention.
            (optionally) tuple: Updated key/value cache if return_kv_cache is True.
        """
        bsz, seq_len, _ = x.size()

        # --- Keys & Values ---
        kv_latent = self.kv_down(x)  # (b, T, d_c)
        if past_kv_latent is not None:
            # (b, T_total, d_c)
            kv_latent = torch.cat([past_kv_latent, kv_latent], dim=1)

        k_c = self.kv_up_k(kv_latent)  # (b, T_total, n_h * h_d)
        v_c = self.kv_up_v(kv_latent)  # (b, T_total, n_h * h_d)
        k_c = k_c.view(bsz, kv_latent.size(1), self.num_heads, self.head_dim)
        v_c = v_c.view(bsz, kv_latent.size(1), self.num_heads, self.head_dim)

        # RoPE component for keys
        # For RoPE, we only need to apply to the current step, but for simplicity, apply to all
        k_r = self.k_rope_proj(x)  # (b, T, h_d)
        if past_kv_latent is not None:
            # For cached steps, need their corresponding input x for RoPE. Assume user provides past_x if needed.
            # For now, just repeat zeros for cached steps (could be improved if past_x is available)
            zeros = torch.zeros(bsz, kv_latent.size(
                1) - seq_len, self.head_dim, device=x.device, dtype=x.dtype)
            k_r = torch.cat([zeros, k_r], dim=1)
        k_r = k_r.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
        k_r = self.rope(k_r)

        # final key: concat compressed + rope
        k = torch.cat([k_c, k_r], dim=-1)

        # --- Queries ---
        q_latent = self.q_down(x)
        q_c = self.q_up(q_latent)
        q_c = q_c.view(bsz, seq_len, self.num_heads, self.head_dim)
        q_r = self.q_rope_proj(x)
        q_r = q_r.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
        q_r = self.rope(q_r)
        q = torch.cat([q_c, q_r], dim=-1)

        # --- Attention scores & output (using F.scaled_dot_product_attention) ---
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v_c.permute(0, 2, 1, 3)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None if mask is None else (mask[:, None, None, :]),
            is_causal=False
        )  # (b, n_h, T, h_d)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(
            bsz, attn_output.size(1), self.num_heads * self.head_dim)

        out = self.out_proj(attn_output)

        if return_kv_cache:
            return out, k_c, k_r, v_c
        else:
            return out
