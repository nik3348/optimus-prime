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


def create_dataloader(dataset, tokenizer, batch_size=8, max_length=512):
    def collate_fn(batch):
        # Combine instruction, context, and response
        texts = []
        for item in batch:
            text = f"Instruction: {item['instruction']}\n"
            if item['context']:
                text += f"Context: {item['context']}\n"
            text += f"Response: {item['response']}"
            texts.append(text)

        encodings = tokenizer(texts,
                              truncation=True,
                              max_length=max_length,
                              padding='max_length',
                              return_tensors='pt')

        # Create causal mask
        seq_len = encodings['input_ids'].size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.unsqueeze(0)  # Add batch dimension
        mask = mask.expand(len(batch), -1, -1)  # Expand to actual batch size

        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': mask,
            'labels': encodings['input_ids'].clone()
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )


def train_epoch(model, dataloader, optimizer, device, scaler, scheduler):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits, _ = model(
                input_ids, attn_mask=attention_mask, training_mode=True)
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Step the scheduler after optimizer step
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits, _ = model(
                    input_ids, attn_mask=attention_mask, training_mode=True)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )

            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize tensorboard
    writer = SummaryWriter('runs/finemath-training')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset(
        "databricks/databricks-dolly-15k",
        split="train",
        num_proc=8
    ).select(range(1000))

    # Create dataloaders
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size]
    )

    train_dataloader = create_dataloader(
        train_dataset,
        tokenizer,
        batch_size=8
    )
    val_dataloader = create_dataloader(
        val_dataset,
        tokenizer,
        batch_size=8
    )

    # Initialize model
    model = Transformer(
        vocab_size=50257,  # GPT-2 tokenizer vocabulary size
        dim=512,
        depth=6,
        num_heads=8,
        num_kv_heads=2,
        checkpointing=True
    ).to(device)

    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=len(train_dataloader) * 3
    )  # 3 epochs

    # Initialize gradient scaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda')

    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(3):  # Train for 3 epochs
        print(f"\nEpoch {epoch + 1}/3")

        # Train
        train_loss = train_epoch(
            model,
            train_dataloader,
            optimizer,
            device,
            scaler,
            scheduler
        )

        # Validate
        val_loss = validate(model, val_dataloader, device)

        # Log metrics to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], epoch)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }
            torch.save(checkpoint, checkpoint_dir / "best_model.pt")
            print(f"Saved checkpoint with validation loss: {val_loss:.4f}")

    # Close tensorboard writer
    writer.close()


if __name__ == "__main__":
    main()
