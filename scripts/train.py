"""Training script for emotion classification model."""

import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.models.emotion_classifier import EmotionClassifier
from src.training.trainer import Trainer
from src.data.fi_dataset import FIDataset
from src.data.fer2013_dataset import FER2013Dataset
from src.data.transforms import get_train_transforms, get_val_transforms


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train emotion classification model")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--dataset-type", type=str, default="fer2013", choices=["fi", "fer2013", "custom"], help="Dataset type")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory for checkpoints")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Backbone architecture (vit_base_patch16_224, resnet50, efficientnet_b0)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Emotion Classification Training")
    print("=" * 70)
    
    # Create configs
    model_config = ModelConfig(backbone=args.backbone)
    training_config = TrainingConfig(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        num_workers=args.num_workers,
    )
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.data_dir}")
    print(f"  Backbone: {args.backbone}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {args.device}")
    
    # Load datasets
    print("\nLoading datasets...")
    
    try:
        train_transform = get_train_transforms(model_config, training_config.use_augmentation)
        val_transform = get_val_transforms(model_config)
        
        if args.dataset_type == "fi":
            # Load FI dataset
            train_dataset = FIDataset.from_directory(
                data_dir=Path(args.data_dir),
                split="train",
                transform=train_transform,
                multi_label=model_config.multi_label,
            )
            
            val_dataset = FIDataset.from_directory(
                data_dir=Path(args.data_dir),
                split="val",
                transform=val_transform,
                multi_label=model_config.multi_label,
            )
            
            print(f"✅ Loaded FI dataset")
            print(f"   Train: {len(train_dataset)} samples")
            print(f"   Val: {len(val_dataset)} samples")
            
        elif args.dataset_type == "fer2013":
            # Load FER2013 dataset
            train_dataset = FER2013Dataset.from_directory(
                data_dir=Path(args.data_dir),
                split="train",
                transform=train_transform,
                system_emotions=model_config.emotions,
            )
            
            val_dataset = FER2013Dataset.from_directory(
                data_dir=Path(args.data_dir),
                split="test",  # FER2013 uses "test" instead of "val"
                transform=val_transform,
                system_emotions=model_config.emotions,
            )
            
            print(f"✅ Loaded FER2013 dataset")
            print(f"   Train: {len(train_dataset)} samples")
            print(f"   Val: {len(val_dataset)} samples")
            
        else:
            print("❌ Custom dataset loading not implemented yet.")
            print("   Please implement your dataset loader in src/data/dataset.py")
            return
            
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        print("\nMake sure your dataset is organized correctly:")
        print("  data_dir/")
        print("    ├── images/")
        print("    └── annotations/")
        print("        ├── train.txt")
        print("        └── val.txt")
        return
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
    )
    
    # Create model
    print("\nCreating model...")
    model = EmotionClassifier(model_config)
    
    model_info = model.get_model_info()
    print(f"✅ Model created")
    print(f"   Backbone: {model_info['backbone']}")
    print(f"   Total params: {model_info['total_params']:,}")
    print(f"   Trainable params: {model_info['trainable_params']:,}")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        training_config=training_config,
        model_config=model_config,
        device=args.device,
    )
    
    # Train
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    trainer.train()
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)
    print(f"Best model saved to: {training_config.output_dir / 'best_model.pth'}")
    print(f"Best validation F1: {trainer.best_val_f1:.4f}")


if __name__ == "__main__":
    main()
