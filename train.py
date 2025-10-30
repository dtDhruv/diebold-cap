"""
Simple training script for GUI Test Automation
Run with: python train.py
"""

import torch
import yaml
import argparse
from pathlib import Path

# Add src to path
import sys
sys.path.append('./src')

from data.dataset import get_dataloaders
from models.blip2_model import GUITestBLIP2
from training.trainer import GUITestTrainer
from utils.evaluation import evaluate_model, GUITestEvaluator


def load_config(config_path: str = './configs/config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    """Main training function."""

    # Load config
    print("Loading configuration...")
    config = load_config(args.config)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")

    # Initialize model
    print("="*60)
    print("Initializing BLIP-2 model...")
    print("="*60)
    model = GUITestBLIP2(
        model_name=config['model']['name'],
        device=device,
        freeze_vision=config['model']['freeze_vision'],
        freeze_qformer=config['model']['freeze_qformer'],
        freeze_lm_encoder=config['model']['freeze_lm_encoder'],
    )

    # Create dataloaders
    print("\n" + "="*60)
    print("Loading dataset...")
    print("="*60)
    train_loader, val_loader = get_dataloaders(
        batch_size=config['dataloader']['batch_size'],
        num_workers=config['dataloader']['num_workers'],
        processor=model.processor,
        device=device,
        max_history_length=config['data']['max_history_length'],
        use_cot=config['data']['use_cot']
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Samples per epoch: {len(train_loader.dataset)}")

    # Create trainer
    print("\n" + "="*60)
    print("Creating trainer...")
    print("="*60)
    trainer = GUITestTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        num_epochs=config['training']['num_epochs'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        max_grad_norm=config['training']['max_grad_norm'],
        warmup_steps=config['training']['warmup_steps'],
        output_dir=config['paths']['output_dir'],
        use_wandb=config['logging']['use_wandb'],
        wandb_project=config['logging']['wandb_project'],
        device=device,
    )

    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    trainer.train()

    # Evaluate
    if args.evaluate:
        print("\n" + "="*60)
        print("Running final evaluation...")
        print("="*60)

        # Load best model
        best_model = GUITestBLIP2(device=device)
        best_model.load_model(f"{config['paths']['output_dir']}/best_model")

        # Evaluate
        evaluator = GUITestEvaluator()
        results = evaluate_model(
            model=best_model,
            dataloader=val_loader,
            evaluator=evaluator,
            max_batches=config['evaluation'].get('max_eval_batches')
        )

        # Save results
        import json
        results_path = Path(config['paths']['output_dir']) / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            # Convert to JSON-serializable format
            json_results = {
                'overall': results['overall'],
                'per_workflow': {k: v for k, v in results['per_workflow'].items()}
            }
            json.dump(json_results, f, indent=2)

        print(f"\n✓ Results saved to {results_path}")

    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GUI Test Automation model")
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Run evaluation after training'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint'
    )

    args = parser.parse_args()
    main(args)
