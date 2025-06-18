import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings


class MLA(nn.Module):
    def __init__(self, nhead, dim_embed, dim_compress, max_seq_len):
        super().__init__()
        self.nhead = nhead
        self.dim_head = dim_embed // nhead
        self.dim_compress = dim_compress

        # Layer norms
        self.norm1 = nn.RMSNorm(dim_compress)
        self.norm2 = nn.RMSNorm(dim_compress)
        self.norm3 = nn.RMSNorm(dim_embed)

        # Linear projections
        self.w_d_kv = nn.Linear(dim_embed, dim_compress, bias=False)
        self.w_u_k = nn.Linear(dim_compress, dim_embed, bias=False)
        self.w_u_v = nn.Linear(dim_compress, dim_embed, bias=False)

        self.w_d_q = nn.Linear(dim_embed, dim_compress, bias=False)
        self.w_u_q = nn.Linear(dim_compress, dim_embed, bias=False)

        # RoPE projections
        self.w_kr = nn.Linear(dim_embed, dim_embed, bias=False)
        self.w_qr = nn.Linear(dim_compress, dim_embed, bias=False)

        # Initialize RoPE
        self.rope = RotaryPositionalEmbeddings(
            dim=self.dim_head, max_seq_len=max_seq_len)

        self.w_o = nn.Linear(dim_embed, dim_embed, bias=False)

        # Initialize absorbed weights as None
        self.absorbed_w_q = None
        self.absorbed_w_o = None

    def compute_absorbed_weights(self):
        """Compute and store the absorbed weights for inference."""
        self.absorbed_w_q = F.linear(self.w_d_q.weight.T, self.w_u_q.weight)
        self.absorbed_w_q = F.linear(self.absorbed_w_q, self.w_u_k.weight.T)
        self.absorbed_w_o = F.linear(self.w_u_v.weight.T, self.w_o.weight)

    def forward(self, h, kv_cache=None, kr_cache=None):
        B, S, _ = h.shape

        # Project input to query and key-value spaces
        cq = self.w_d_q(h)
        ckv = self.w_d_kv(h)

        # Layer Norm
        cq = self.norm1(cq)
        ckv = self.norm2(ckv)

        # Prepare RoPE inputs
        qr = self.w_qr(cq)
        kr = self.w_kr(h)

        # Reshape for RoPE application to match expected format [b, s, n_h, h_d]
        qr = qr.view(B, S, self.nhead, self.dim_head)
        kr = kr.view(B, S, self.nhead, self.dim_head)

        # Apply RoPE - input and output shapes are the same [b, s, n_h, h_d]
        qr = self.rope(qr).view(B, S, -1)
        kr = self.rope(kr).view(B, S, -1)

        if self.training:
            # Project to attention space
            qc = self.w_u_q(cq)
            kc = self.w_u_k(ckv)
            v = self.w_u_v(ckv)

            # Combine with RoPE
            q = torch.cat([qc, qr], dim=-1)
            k = torch.cat([kc, kr], dim=-1)

            # Compute attention
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            u = self.w_o(o)

        else:
            # Use pre-computed absorbed weights if available, otherwise compute them
            if self.absorbed_w_q is None or self.absorbed_w_o is None:
                self.compute_absorbed_weights()

            absorbed_q = F.linear(h, self.absorbed_w_q.T)

            if kv_cache is not None:
                ckv = torch.cat([kv_cache, ckv], dim=1)

            if kr_cache is not None:
                kr = torch.cat([kr_cache, kr], dim=1)

            # Compute attention scores
            attn_scores = torch.matmul(absorbed_q, ckv.transpose(-2, -1))
            attn_scores_rope = torch.matmul(qr, kr.transpose(-2, -1))

            # Combine scores and compute attention weights
            attn_weights = torch.softmax(
                (attn_scores + attn_scores_rope)/((self.nhead + self.dim_head) ** 0.5), dim=-1)

            # Compute output
            o = torch.matmul(attn_weights, ckv)
            u = F.linear(o, self.absorbed_w_o.T)

        # Post-attention norm
        u = self.norm3(u)
        return u, ckv, kr
