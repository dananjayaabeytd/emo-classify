"""Training script for emotion classification model."""

import argparse
import torch
from pathlib import Path

from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.models.emotion_classifier import EmotionClassifier
from src.training.trainer import Trainer
from src.data.dataset import EmotionDataset
from src.data.dataloader import create_dataloaders


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train emotion classification model")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory for checkpoints")
    parser.add_argument("--backbone", type=str, default="vit_base_patch16_224", help="Backbone architecture")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Create configs
    model_config = ModelConfig(backbone=args.backbone)
    training_config = TrainingConfig(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
    )
    
    # Load dataset (you'll need to implement this based on your dataset format)
    # For now, this is a placeholder
    print("Loading dataset...")
    # dataset = EmotionDataset.from_directory(...)
    
    # Create dataloaders
    # train_loader, val_loader, test_loader = create_dataloaders(
    #     dataset, training_config, model_config
    # )
    
    # Create model
    print("Creating model...")
    model = EmotionClassifier(model_config)
    
    print(f"Model info: {model.get_model_info()}")
    
    # Create trainer
    # trainer = Trainer(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     training_config=training_config,
    #     model_config=model_config,
    #     device=args.device,
    # )
    
    # Train
    # print("Starting training...")
    # trainer.train()
    
    print("Training script ready! Implement dataset loading for your specific dataset.")


if __name__ == "__main__":
    main()
