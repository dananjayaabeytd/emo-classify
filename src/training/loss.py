"""Loss functions for emotion classification."""

import torch
import torch.nn as nn
from typing import Callable


def get_loss_function(multi_label: bool = True) -> Callable:
    """
    Get appropriate loss function based on classification type.

    Args:
        multi_label: Whether this is multi-label classification

    Returns:
        Loss function
    """
    if multi_label:
        # Binary Cross Entropy with Logits for multi-label
        return nn.BCEWithLogitsLoss()
    else:
        # Cross Entropy for single-label
        return nn.CrossEntropyLoss()


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Reference: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor in [0, 1]
            gamma: Focusing parameter >= 0
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predictions (logits)
            targets: Ground truth labels

        Returns:
            Loss value
        """
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCEWithLogitsLoss(nn.Module):
    """Weighted BCE loss for handling class imbalance in multi-label classification."""

    def __init__(self, pos_weights: torch.Tensor = None):
        """
        Initialize weighted BCE loss.

        Args:
            pos_weights: Positive class weights for each class
        """
        super().__init__()
        self.pos_weights = pos_weights

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted BCE loss.

        Args:
            inputs: Predictions (logits)
            targets: Ground truth labels

        Returns:
            Loss value
        """
        return nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weights
        )
