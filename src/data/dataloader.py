"""DataLoader creation utilities."""

from typing import Tuple
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from .dataset import EmotionDataset
from .transforms import get_train_transforms, get_val_transforms
from ..config.model_config import ModelConfig
from ..config.training_config import TrainingConfig


def create_dataloaders(
    dataset: EmotionDataset,
    training_config: TrainingConfig,
    model_config: ModelConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders from a dataset.

    Args:
        dataset: The full emotion dataset
        training_config: Training configuration
        model_config: Model configuration

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(training_config.train_split * total_size)
    val_size = int(training_config.val_split * total_size)
    test_size = total_size - train_size - val_size

    # Set random seed for reproducibility
    torch.manual_seed(training_config.seed)

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Apply different transforms to each split
    train_transform = get_train_transforms(model_config, training_config.use_augmentation)
    val_transform = get_val_transforms(model_config)

    # Update transforms (note: random_split creates Subset objects)
    # We'll need to handle this differently - create separate dataset instances
    # For now, this is a simplified version

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
    )

    return train_loader, val_loader, test_loader


def create_inference_dataloader(
    dataset: EmotionDataset,
    batch_size: int = 32,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create a dataloader for inference.

    Args:
        dataset: Emotion dataset
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        DataLoader for inference
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
