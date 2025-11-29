"""Image transformations for training and validation."""

from typing import Callable

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from ..config.model_config import ModelConfig


def get_train_transforms(config: ModelConfig, use_augmentation: bool = True) -> Callable:
    """
    Get training transforms with optional augmentation.

    Args:
        config: Model configuration
        use_augmentation: Whether to apply data augmentation

    Returns:
        Composed transforms
    """
    transforms = [
        T.Resize((config.image_size, config.image_size), interpolation=InterpolationMode.BICUBIC),
    ]

    if use_augmentation:
        transforms.extend(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                ),
                T.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                ),
            ]
        )

    transforms.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=config.mean, std=config.std),
        ]
    )

    return T.Compose(transforms)


def get_val_transforms(config: ModelConfig) -> Callable:
    """
    Get validation/test transforms (no augmentation).

    Args:
        config: Model configuration

    Returns:
        Composed transforms
    """
    transforms = [
        T.Resize((config.image_size, config.image_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=config.mean, std=config.std),
    ]

    return T.Compose(transforms)


def get_inference_transforms(config: ModelConfig) -> Callable:
    """
    Get inference transforms (same as validation).

    Args:
        config: Model configuration

    Returns:
        Composed transforms
    """
    return get_val_transforms(config)
