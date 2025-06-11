import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from transformers import AutoTokenizer
from model import Transformer
from tqdm import tqdm
from pathlib import Path
from config import ModelConfig, TrainingConfig


# Constants
PAD_TOKEN = "eos_token"
IGNORE_INDEX = -100


def create_dataloader(dataset, tokenizer, config: TrainingConfig) -> DataLoader:
    """Create a DataLoader for the given dataset with proper batching and masking.

    Args:
        dataset: The dataset to create a DataLoader for
        tokenizer: The tokenizer to use for encoding
        config: Training configuration

    Returns:
        DataLoader configured for training
    """
    def collate_fn(batch):
        try:
            # Combine instruction, context, and response
            texts = []
            for item in batch:
                text = f"Instruction: {item['instruction']}\n"
                if item['context']:
                    text += f"Context: {item['context']}\n"
                text += f"Response: {item['response']}"
                texts.append(text)

            encodings = tokenizer(
                texts,
                truncation=True,
                max_length=config.max_length,
                padding='max_length',
                return_tensors='pt'
            )

            # Create causal mask
            seq_len = encodings['input_ids'].size(1)
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            mask = mask.unsqueeze(0).expand(len(batch), -1, -1)

            # Ensure all tensors are contiguous and properly cloned
            input_ids = encodings['input_ids'].contiguous()
            attention_mask = mask.contiguous()
            labels = input_ids.clone().contiguous()

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
        except Exception as e:
            print(f"Error in collate_fn: {str(e)}")
            raise

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True  # Enable faster data transfer to GPU
    )


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    config: TrainingConfig,
    writer: SummaryWriter,
    epoch: int
) -> float:
    """Train the model for one epoch.

    Args:
        model: The model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer for training
        device: Device to train on
        scaler: Gradient scaler for mixed precision training
        config: Training configuration
        writer: TensorBoard writer for logging
        epoch: Current epoch number

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    # Add a progress bar for training
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training")
    for batch_idx, batch in enumerate(progress_bar):
        try:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits, _ = model(input_ids, attn_mask=attention_mask)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=IGNORE_INDEX
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_val)
            scaler.step(optimizer)
            scaler.update()

            current_loss = loss.item()
            total_loss += current_loss

            # Log batch-level training loss, learning rate, and update progress bar
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar('Loss/train_batch', current_loss, global_step)
            writer.add_scalar(
                'Learning_rate', optimizer.param_groups[0]['lr'], global_step)
            progress_bar.set_postfix(
                loss=f"{current_loss:.4f}", avg_loss=f"{total_loss/(batch_idx+1):.4f}")

        except Exception as e:
            print(f"Error during training step (batch {batch_idx}): {str(e)}")
            continue

    avg_train_loss = total_loss / num_batches
    writer.add_scalar('Loss/train_epoch_avg', avg_train_loss, epoch)
    return avg_train_loss


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int
) -> float:
    """Validate the model.

    Args:
        model: The model to validate
        dataloader: DataLoader for validation data
        device: Device to validate on
        writer: TensorBoard writer for logging
        epoch: Current epoch number

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)

    # Add a progress bar for validation
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} Validation")
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits, _ = model(input_ids, attn_mask=attention_mask)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=IGNORE_INDEX
                    )

                current_loss = loss.item()
                total_loss += current_loss

                # Log batch-level validation loss and update progress bar
                global_step = epoch * num_batches + batch_idx
                writer.add_scalar('Loss/val_batch', current_loss, global_step)
                progress_bar.set_postfix(
                    loss=f"{current_loss:.4f}", avg_loss=f"{total_loss/(batch_idx+1):.4f}")

            except Exception as e:
                print(
                    f"Error during validation step (batch {batch_idx}): {str(e)}")
                continue

    avg_val_loss = total_loss / num_batches
    writer.add_scalar('Loss/val_epoch_avg', avg_val_loss, epoch)
    return avg_val_loss


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    val_loss: float,
    config: TrainingConfig,
    is_best: bool = False
) -> None:
    """Save a training checkpoint.

    Args:
        model: The model to save
        optimizer: The optimizer state to save
        scheduler: The scheduler state to save
        epoch: Current epoch number
        val_loss: Current validation loss
        config: Training configuration
        is_best: Whether this is the best model so far
    """
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'config': config
    }

    # Save latest checkpoint
    latest_path = checkpoint_dir / "latest_model.pt"
    torch.save(checkpoint, latest_path)
    print(f"Saved latest checkpoint to {latest_path}")

    # Save best checkpoint if this is the best model
    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
        print(
            f"Saved best checkpoint to {best_path} (Validation Loss: {val_loss:.4f})")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: TrainingConfig
) -> tuple[int, float]:
    """Load the latest checkpoint if it exists.

    Args:
        model: The model to load weights into
        optimizer: The optimizer to load state into
        scheduler: The scheduler to load state into
        config: Training configuration

    Returns:
        Tuple of (last_epoch, best_val_loss)
    """
    checkpoint_path = Path(config.checkpoint_dir) / "latest_model.pt"
    if not checkpoint_path.exists():
        print("No checkpoint found. Starting from scratch.")
        return 0, float('inf')

    print(f"Loading checkpoint from {checkpoint_path}")
    try:
        # Load to CPU first to avoid device issues
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        print("Could not load checkpoint. Starting from scratch.")
        return 0, float('inf')

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state if available
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state if available
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Get epoch and validation loss
    last_epoch = checkpoint.get('epoch', 0)
    best_val_loss = checkpoint.get('val_loss', float('inf'))

    print(
        f"Successfully resumed from epoch {last_epoch + 1} with best validation loss: {best_val_loss:.4f}")
    return last_epoch + 1, best_val_loss


def main():
    # Initialize configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize tensorboard
    writer = SummaryWriter(training_config.tensorboard_dir)
    print(
        f"TensorBoard logs will be saved to: {training_config.tensorboard_dir}")

    # Initialize model
    model = Transformer(
        config=model_config,
        mlp_ratio=4,
        dropout=0.0
    ).to(device)

    # Print model parameters
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print("\n--- Model Information ---")
    print("Model: Transformer")
    print(f"Total trainable parameters: {total_params:,}")
    print("Model Configuration:")
    print(f"  Vocabulary Size: {model_config.vocab_size}")
    print("-------------------------")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Ensure pad_token is set for consistency in DataLoader
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer's pad_token set to eos_token: {tokenizer.pad_token}")
    else:
        print(f"Tokenizer's pad_token: {tokenizer.pad_token}")

    # Load dataset
    print(f"\nLoading dataset: {training_config.dataset_name}")
    try:
        dataset = load_dataset(
            training_config.dataset_name,
            split="train",
            num_proc=training_config.num_workers
        )
        if training_config.dataset_size > 0:
            dataset = dataset.select(
                range(min(training_config.dataset_size, len(dataset))))
        print(f"Total dataset size: {len(dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        print("Exiting training.")
        return

    # Create dataloaders
    train_size = int(training_config.train_val_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size]
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_dataloader = create_dataloader(
        train_dataset, tokenizer, training_config)
    val_dataloader = create_dataloader(val_dataset, tokenizer, training_config)
    print(f"Training batches per epoch: {len(train_dataloader)}")
    print(f"Validation batches per epoch: {len(val_dataloader)}")

    # Initialize optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )
    # T_max should be the total number of training steps (batches * epochs)
    total_training_steps = len(train_dataloader) * training_config.num_epochs
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_training_steps
    )
    scaler = torch.amp.GradScaler('cuda')

    print("\n--- Training Configuration ---")
    print(f"Learning Rate: {training_config.learning_rate}")
    print(f"Weight Decay: {training_config.weight_decay}")
    print(f"Batch Size: {training_config.batch_size}")
    print(f"Number of Epochs: {training_config.num_epochs}")
    print(f"Gradient Clip Value: {training_config.gradient_clip_val}")
    print("Mixed Precision (AMP): Enabled (float16)")
    print(f"Checkpoint Directory: {training_config.checkpoint_dir}")
    print("-----------------------------")

    # Load checkpoint if exists
    start_epoch, best_val_loss = load_checkpoint(
        model,
        optimizer,
        scheduler,
        training_config
    )

    # Training loop
    for epoch in range(start_epoch, training_config.num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{training_config.num_epochs} ---")

        # Train
        train_loss = train_epoch(
            model,
            train_dataloader,
            optimizer,
            device,
            scaler,
            training_config,
            writer,
            epoch
        )
        scheduler.step()

        print(f"Epoch {epoch + 1} Average Training Loss: {train_loss:.4f}")
        writer.add_scalar('Loss/epoch_avg_train', train_loss, epoch)

        # Validate
        val_loss = validate(model, val_dataloader, device, writer, epoch)
        print(f"Epoch {epoch + 1} Average Validation Loss: {val_loss:.4f}")
        writer.add_scalar('Loss/epoch_avg_val', val_loss, epoch)

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            print(
                f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving best model.")
            best_val_loss = val_loss
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            val_loss,
            training_config,
            is_best
        )
        print(f"Current best validation loss: {best_val_loss:.4f}")

    writer.close()
    print("\n--- Training completed! ---")
    print(f"Final best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
