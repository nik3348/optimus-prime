import torch
from model import Transformer

if __name__ == "__main__":
    model = Transformer(
        vocab_size=32000,
        dim=512,
        depth=6,
        num_heads=8,
        num_kv_heads=2,
        checkpointing=True
    )

    input_ids = torch.randint(0, 32000, (2, 128))

    with torch.autocast('cuda'):
        # During training, we don't need KV cache computation
        logits, _ = model(input_ids, training_mode=True)

    print(logits.shape)
