import torch
from transformers import GPT2Tokenizer
from pathlib import Path
import logging
from main import ModelConfig, DecoderOnlyTransformer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_model(checkpoint_path, device='cuda'):
    """Load the model from checkpoint"""
    config = ModelConfig()
    model = DecoderOnlyTransformer(config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, top_p=0.9, device='cuda'):
    """Generate text using the model"""
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # Initialize KV cache
    kv_cache = None

    # Generate tokens
    generated_ids = input_ids
    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions
            logits, kv_cache = model(generated_ids, kv_cache)

            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature

            # Apply top-p (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(
                next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[...,
                                     1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Stop if we generate an EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode and return the generated text
    generated_text = tokenizer.decode(
        generated_ids[0], skip_special_tokens=True)
    return generated_text


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2',
                                              model_max_length=2048,
                                              padding_side='right')
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    checkpoint_path = Path('checkpoints/best_model.pt')
    if not checkpoint_path.exists():
        logger.error(f"No model checkpoint found at {checkpoint_path}")
        return

    model = load_model(checkpoint_path, device)
    logger.info("Model loaded successfully")

    # Interactive generation loop
    print("\nEnter your prompt (type 'quit' to exit):")
    while True:
        prompt = input("\nPrompt: ").strip()
        if prompt.lower() == 'quit':
            break

        if not prompt:
            continue

        try:
            generated_text = generate_text(
                model,
                tokenizer,
                prompt,
                max_length=100,
                temperature=0.7,
                top_p=0.9,
                device=device
            )
            print("\nGenerated text:")
            print(generated_text)
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")


if __name__ == "__main__":
    main()
