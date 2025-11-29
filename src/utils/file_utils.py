"""File utilities for saving and loading models."""

import torch
from pathlib import Path
from typing import Dict, Any

from ..models.emotion_classifier import EmotionClassifier
from ..config.model_config import ModelConfig


def save_model(
    model: EmotionClassifier,
    save_path: Path,
    epoch: int = None,
    optimizer_state: Dict = None,
    metrics: Dict[str, float] = None,
    **kwargs,
):
    """
    Save model checkpoint with metadata.

    Args:
        model: The model to save
        save_path: Path to save the checkpoint
        epoch: Current epoch number
        optimizer_state: Optimizer state dict
        metrics: Dictionary of metrics
        **kwargs: Additional metadata to save
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model.config,
    }

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if optimizer_state is not None:
        checkpoint["optimizer_state_dict"] = optimizer_state

    if metrics is not None:
        checkpoint["metrics"] = metrics

    # Add any additional kwargs
    checkpoint.update(kwargs)

    # Ensure directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


def load_model(
    checkpoint_path: Path,
    device: str = "cpu",
) -> tuple[EmotionClassifier, Dict[str, Any]]:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on

    Returns:
        Tuple of (model, metadata)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model config
    model_config = checkpoint.get("model_config")
    if model_config is None:
        raise ValueError("Model config not found in checkpoint")

    # Create model
    model = EmotionClassifier(model_config)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Extract metadata
    metadata = {
        k: v for k, v in checkpoint.items() if k not in ["model_state_dict", "model_config"]
    }

    print(f"Model loaded from {checkpoint_path}")
    if "epoch" in metadata:
        print(f"  Epoch: {metadata['epoch']}")
    if "metrics" in metadata:
        print(f"  Metrics: {metadata['metrics']}")

    return model, metadata


def save_predictions(
    predictions: list,
    save_path: Path,
    format: str = "json",
):
    """
    Save predictions to file.

    Args:
        predictions: List of prediction dictionaries
        save_path: Path to save file
        format: 'json' or 'csv'
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        import json

        with open(save_path, "w") as f:
            json.dump(predictions, f, indent=2)

    elif format == "csv":
        import pandas as pd

        df = pd.DataFrame(predictions)
        df.to_csv(save_path, index=False)

    else:
        raise ValueError(f"Unknown format: {format}")

    print(f"Predictions saved to {save_path}")
