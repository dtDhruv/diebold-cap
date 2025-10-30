"""
Training script for GUI Test Automation model
Optimized for Colab/Kaggle with memory-efficient training
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import wandb
from typing import Dict, Optional
import os
from pathlib import Path


class GUITestTrainer:
    """
    Trainer for BLIP-2 GUI test automation model.

    Features:
    - Mixed precision training (fp16)
    - Gradient accumulation
    - Gradient checkpointing
    - Learning rate scheduling
    - WandB logging
    - Automatic checkpointing
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        num_epochs: int = 3,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 100,
        output_dir: str = "./outputs",
        use_wandb: bool = True,
        wandb_project: str = "gui-test-automation",
        device: str = "cuda",
    ):
        """
        Args:
            model: GUITestBLIP2 model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            learning_rate: Learning rate
            weight_decay: Weight decay for AdamW
            num_epochs: Number of training epochs
            gradient_accumulation_steps: Accumulate gradients over N steps
            max_grad_norm: Maximum gradient norm for clipping
            warmup_steps: Number of warmup steps
            output_dir: Directory to save checkpoints
            use_wandb: Whether to use Weights & Biases logging
            wandb_project: WandB project name
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model.model, 'gradient_checkpointing_enable'):
            self.model.model.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled")

        # Optimizer
        self.optimizer = AdamW(
            [p for p in self.model.model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # Learning rate scheduler
        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Mixed precision scaler
        self.scaler = GradScaler()

        # WandB
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project=wandb_project,
                config={
                    "learning_rate": learning_rate,
                    "epochs": num_epochs,
                    "batch_size": train_loader.batch_size,
                    "gradient_accumulation": gradient_accumulation_steps,
                    "model": "BLIP-2 FLAN-T5-base",
                },
            )

        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.model.train()

        total_loss = 0
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.num_epochs}",
        )

        self.optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            # Forward pass with mixed precision
            with autocast():
                outputs = self.model(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    labels=batch['labels'],
                )
                loss = outputs.loss

                # Normalize loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Update weights every N steps
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.model.parameters(),
                    self.max_grad_norm,
                )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

            # Logging
            total_loss += loss.item() * self.gradient_accumulation_steps
            avg_loss = total_loss / (step + 1)

            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })

            if self.use_wandb and self.global_step % 10 == 0:
                wandb.log({
                    "train/loss": loss.item() * self.gradient_accumulation_steps,
                    "train/learning_rate": self.scheduler.get_last_lr()[0],
                    "train/step": self.global_step,
                })

        return {"train_loss": avg_loss}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.model.eval()

        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(self.val_loader, desc="Validating")

        for batch in progress_bar:
            with autocast():
                outputs = self.model(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    labels=batch['labels'],
                )
                loss = outputs.loss

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({'val_loss': f'{total_loss/num_batches:.4f}'})

        avg_loss = total_loss / num_batches

        return {"val_loss": avg_loss}

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / f"checkpoint-epoch-{epoch+1}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_model(str(checkpoint_dir))

        # Save training state
        torch.save({
            'epoch': epoch,
            'global_step': self.global_step,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'metrics': metrics,
        }, checkpoint_dir / "training_state.pt")

        print(f"✓ Checkpoint saved to {checkpoint_dir}")

        # Save best model
        if is_best:
            best_dir = self.output_dir / "best_model"
            best_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_model(str(best_dir))
            print(f"✓ Best model saved to {best_dir}")

    def train(self):
        """Full training loop."""
        print(f"\n{'='*60}")
        print("Starting training")
        print(f"{'='*60}\n")

        for epoch in range(self.num_epochs):
            print(f"\n--- Epoch {epoch+1}/{self.num_epochs} ---")

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Combine metrics
            metrics = {**train_metrics, **val_metrics}

            # Log to WandB
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    **metrics,
                })

            # Print metrics
            print(f"\nEpoch {epoch+1} results:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")

            # Save checkpoint
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
                print(f"\n✓ New best validation loss: {self.best_val_loss:.4f}")

            self.save_checkpoint(epoch, metrics, is_best)

        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")

        if self.use_wandb:
            wandb.finish()


def train_model(
    model,
    train_loader,
    val_loader,
    config: Dict,
    output_dir: str = "./outputs",
):
    """
    Convenience function to train model.

    Args:
        model: GUITestBLIP2 model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        config: Training configuration dict
        output_dir: Output directory

    Returns:
        Trained model
    """
    trainer = GUITestTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        **config,
    )

    trainer.train()

    return model


if __name__ == "__main__":
    print("Trainer module loaded successfully")
    print("Use train_model() function to start training")
