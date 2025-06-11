import torch
from model import Transformer

# Example usage
if __name__ == "__main__":
    model = Transformer(
        vocab_size=32000,
        dim=512,
        depth=6,
        num_heads=8,
        checkpointing=True
    )
    input_ids = torch.randint(0, 32000, (2, 128))

    with torch.autocast('cuda'):
        logits = model(input_ids)

    print(logits.shape)
