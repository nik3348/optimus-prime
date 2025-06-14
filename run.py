import torch
from transformers import AutoTokenizer
from model import Transformer
from config import ModelConfig, TrainingConfig


def load_model(checkpoint_path: str, device: torch.device) -> tuple[Transformer, AutoTokenizer]:
    """Load the model and tokenizer from checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load the model on

    Returns:
        Tuple of (model, tokenizer)
    """
    # Load configuration and initialize model
    model_config = ModelConfig()
    model = Transformer(config=model_config).to(device)

    # Load checkpoint
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_text(
    model: Transformer,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: torch.device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
) -> str:
    """Generate text from the model given a prompt, using KV and KR caching for efficient decoding.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: Input prompt text
        max_length: Maximum length of generated text
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        device: Device to run generation on

    Returns:
        Generated text
    """
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # Initialize caches
    kv_caches = None
    kr_caches = None
    generated = input_ids

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for _ in range(max_length):
                # Pass None for caches on first step, then use cached values for subsequent steps
                if kv_caches is None:
                    logits, kv_caches, kr_caches = model(generated)
                else:
                    logits, kv_caches, kr_caches = model(
                        generated[:, -1:],
                        kv_caches=kv_caches,
                        kr_caches=kr_caches
                    )

                next_token_logits = logits[:, -1, :] / temperature

                # Apply top-p sampling
                probs = torch.softmax(next_token_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(
                    probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[...,
                                         1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)

                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

                # Stop if we generate an EOS token
                if next_token.item() == tokenizer.eos_token_id:
                    break

    # Decode and return the generated text
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated_text


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize configurations
    training_config = TrainingConfig()

    # Load model and tokenizer
    try:
        model, tokenizer = load_model(
            training_config.checkpoint_dir + "/best_model.pt", device)
        print(
            f"Successfully loaded model from {training_config.checkpoint_dir}/best_model.pt")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    print("\nModel loaded successfully! You can now start generating text.")
    print("Type 'quit' to exit.")
    print("-" * 50)

    while True:
        try:
            # Get user input
            prompt = input("\nEnter your prompt: ").strip()
            if prompt.lower() == 'quit':
                break

            if not prompt:
                print("Please enter a non-empty prompt.")
                continue

            # # Get user input for instruction
            # instruction = input("\nEnter instruction: ").strip()
            # if instruction.lower() == 'quit':
            #     break
            # if not instruction:
            #     print("Please enter a non-empty instruction.")
            #     continue

            # # Get user input for context (optional)
            # context = input("Enter context (optional, press Enter to skip): ").strip()

            # # Construct prompt in the same format as training
            # prompt = f"Instruction: {instruction}\n"
            # if context:
            #     prompt += f"Context: {context}\n"
            # prompt += "Response: "

            # Generate text
            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device
            )

            # Print the generated text
            print("\nGenerated text:")
            print("-" * 50)
            print(generated_text)
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            continue


if __name__ == "__main__":
    main()
