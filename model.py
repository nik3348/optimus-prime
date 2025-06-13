import torch.nn as nn
from config import ModelConfig
from src.transformer import MLATransformerBlock


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig, mlp_ratio=4):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.layers = nn.ModuleList([
            MLATransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.norm = nn.RMSNorm(config.embedding_dim)
        self.head = nn.Linear(
            config.embedding_dim, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, past_kv_latents=None, return_kv_cache=False, mask=None):
        """
        Forward pass with optional KV caching for incremental decoding.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            past_kv_latents: List of cached KV latents for each layer, or None
            return_kv_cache: Whether to return updated KV caches
            mask: Optional attention mask
        Returns:
            If return_kv_cache: (output, new_kv_latents)
            else: output
        """
        # If doing incremental decoding (input_ids is shape [B, 1]), only embed last token
        if past_kv_latents is not None and all(kv is not None for kv in past_kv_latents):
            # Only decode last token
            x = self.embed(input_ids[:, -1:])
        else:
            x = self.embed(input_ids)

        new_kv_latents = [] if return_kv_cache else None
        if past_kv_latents is None:
            past_kv_latents = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            # Each layer expects either None or a 3D tensor (not a list)
            layer_past = past_kv_latents[i] if past_kv_latents is not None else None
            if return_kv_cache:
                x, new_kv = layer(
                    x, mask=mask, past_kv_latent=layer_past, return_kv_cache=True)
                new_kv_latents.append(new_kv)
            else:
                x = layer(
                    x, mask=mask, past_kv_latent=layer_past, return_kv_cache=False)
        x = self.head(self.norm(x))
        if return_kv_cache:
            return x, new_kv_latents
        else:
            return x
