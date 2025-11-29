"""Emotion classification model architecture."""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from .backbone import create_backbone
from ..config.model_config import ModelConfig


class EmotionClassifier(nn.Module):
    """
    Emotion classification model with pretrained backbone.

    Supports both single-label and multi-label classification.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the emotion classifier.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Create backbone
        self.backbone, feature_dim = create_backbone(
            config.backbone,
            pretrained=config.pretrained,
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(feature_dim, config.num_classes),
        )

        # Initialize weights for the head
        self._init_head()

    def _init_head(self):
        """Initialize the classification head weights."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image tensor of shape (batch_size, 3, H, W)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Extract features
        features = self.backbone(x)

        # Classification
        logits = self.head(features)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions.

        Args:
            x: Input image tensor

        Returns:
            Probability tensor (after sigmoid for multi-label)
        """
        logits = self.forward(x)

        if self.config.multi_label:
            # Multi-label: apply sigmoid
            probs = torch.sigmoid(logits)
        else:
            # Single-label: apply softmax
            probs = torch.softmax(logits, dim=-1)

        return probs

    def predict_emotions(
        self,
        x: torch.Tensor,
        threshold: float = 0.35,
        top_k: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict emotions from images.

        Args:
            x: Input image tensor
            threshold: Threshold for multi-label classification
            top_k: If set, return top K emotions (for single-label or top-k multi-label)

        Returns:
            Tuple of (predicted_labels, probabilities)
        """
        probs = self.predict_proba(x)

        if self.config.multi_label:
            # Multi-label: threshold-based
            if top_k is not None:
                # Get top-k emotions
                top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
                predictions = torch.zeros_like(probs)
                predictions.scatter_(1, top_indices, 1.0)
            else:
                # Threshold-based
                predictions = (probs > threshold).float()

                # If no emotion passes threshold, take the top one
                no_predictions = predictions.sum(dim=-1) == 0
                if no_predictions.any():
                    max_indices = probs.argmax(dim=-1)
                    predictions[no_predictions, max_indices[no_predictions]] = 1.0
        else:
            # Single-label: argmax
            predictions = torch.zeros_like(probs)
            max_indices = probs.argmax(dim=-1)
            predictions[torch.arange(predictions.size(0)), max_indices] = 1.0

        return predictions, probs

    def get_model_info(self) -> Dict[str, any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "backbone": self.config.backbone,
            "num_classes": self.config.num_classes,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "multi_label": self.config.multi_label,
        }

    def freeze_backbone(self):
        """Freeze backbone parameters for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_layers(self, num_layers: int):
        """
        Freeze the first N layers of the backbone.

        Args:
            num_layers: Number of layers to freeze
        """
        # This is backbone-specific and would need custom implementation
        # for different architectures
        raise NotImplementedError("Layer-wise freezing not yet implemented")
