import torch.nn as nn
from config import ModelConfig
from src.transformer import TransformerBlock, MLATransformerBlock


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

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))
