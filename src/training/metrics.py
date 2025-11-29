"""Metrics for emotion classification evaluation."""

import torch
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
)
from typing import Dict, List


class EmotionMetrics:
    """Calculate and track metrics for emotion classification."""

    def __init__(self, num_classes: int = 8, emotion_labels: List[str] = None):
        """
        Initialize metrics tracker.

        Args:
            num_classes: Number of emotion classes
            emotion_labels: List of emotion label names
        """
        self.num_classes = num_classes
        self.emotion_labels = emotion_labels or [f"emotion_{i}" for i in range(num_classes)]

        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with new predictions.

        Args:
            predictions: Predicted labels (binary or probabilities after threshold)
            targets: Ground truth labels
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        self.predictions.append(predictions)
        self.targets.append(targets)

    def compute(self, multi_label: bool = True) -> Dict[str, float]:
        """
        Compute all metrics.

        Args:
            multi_label: Whether this is multi-label classification

        Returns:
            Dictionary of metric names and values
        """
        if not self.predictions:
            return {}

        # Concatenate all batches
        predictions = np.vstack(self.predictions)
        targets = np.vstack(self.targets)

        metrics = {}

        if multi_label:
            # Multi-label metrics
            metrics["accuracy"] = accuracy_score(targets, predictions)
            metrics["f1_micro"] = f1_score(targets, predictions, average="micro", zero_division=0)
            metrics["f1_macro"] = f1_score(targets, predictions, average="macro", zero_division=0)
            metrics["f1_weighted"] = f1_score(
                targets, predictions, average="weighted", zero_division=0
            )
            metrics["precision_macro"] = precision_score(
                targets, predictions, average="macro", zero_division=0
            )
            metrics["recall_macro"] = recall_score(
                targets, predictions, average="macro", zero_division=0
            )

            # Per-class F1 scores
            per_class_f1 = f1_score(targets, predictions, average=None, zero_division=0)
            for i, label in enumerate(self.emotion_labels):
                metrics[f"f1_{label}"] = per_class_f1[i]

        else:
            # Single-label metrics
            predictions_flat = predictions.argmax(axis=1)
            targets_flat = targets.argmax(axis=1)

            metrics["accuracy"] = accuracy_score(targets_flat, predictions_flat)
            metrics["f1_micro"] = f1_score(
                targets_flat, predictions_flat, average="micro", zero_division=0
            )
            metrics["f1_macro"] = f1_score(
                targets_flat, predictions_flat, average="macro", zero_division=0
            )
            metrics["f1_weighted"] = f1_score(
                targets_flat, predictions_flat, average="weighted", zero_division=0
            )
            metrics["precision_macro"] = precision_score(
                targets_flat, predictions_flat, average="macro", zero_division=0
            )
            metrics["recall_macro"] = recall_score(
                targets_flat, predictions_flat, average="macro", zero_division=0
            )

            # Per-class F1 scores
            per_class_f1 = f1_score(
                targets_flat, predictions_flat, average=None, zero_division=0
            )
            for i, label in enumerate(self.emotion_labels):
                metrics[f"f1_{label}"] = per_class_f1[i]

        return metrics

    def get_classification_report(self, multi_label: bool = True) -> str:
        """
        Get detailed classification report.

        Args:
            multi_label: Whether this is multi-label classification

        Returns:
            Classification report string
        """
        if not self.predictions:
            return "No predictions available"

        predictions = np.vstack(self.predictions)
        targets = np.vstack(self.targets)

        if multi_label:
            return classification_report(
                targets,
                predictions,
                target_names=self.emotion_labels,
                zero_division=0,
            )
        else:
            predictions_flat = predictions.argmax(axis=1)
            targets_flat = targets.argmax(axis=1)
            return classification_report(
                targets_flat,
                predictions_flat,
                target_names=self.emotion_labels,
                zero_division=0,
            )
