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

    def forward(self, input_ids, kv_caches=None, kr_caches=None):
        """
        Forward pass with optional KV and KR caching for incremental decoding.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            kv_caches: List of cached KV tensors for each layer, or None
            kr_caches: List of cached KR tensors for each layer, or None
        Returns:
            output: Model output logits
            kv_caches: Updated KV caches for each layer
            kr_caches: Updated KR caches for each layer
        """
        # If doing incremental decoding (input_ids is shape [B, 1]), only embed last token
        if kv_caches is not None and all(kv is not None for kv in kv_caches):
            # Only decode last token
            x = self.embed(input_ids[:, -1:])
        else:
            x = self.embed(input_ids)

        new_kv_caches = []
        new_kr_caches = []

        if kv_caches is None:
            kv_caches = [None] * len(self.layers)
        if kr_caches is None:
            kr_caches = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            x, kv_cache, kr_cache = layer(
                x,
                kv_cache=kv_caches[i],
                kr_cache=kr_caches[i]
            )
            new_kv_caches.append(kv_cache)
            new_kr_caches.append(kr_cache)

        x = self.head(self.norm(x))
        return x, new_kv_caches, new_kr_caches
