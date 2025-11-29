"""Backbone model creation utilities."""

import timm
import torch.nn as nn
from typing import Tuple


def create_backbone(
    backbone_name: str,
    pretrained: bool = True,
    num_classes: int = 8,
) -> Tuple[nn.Module, int]:
    """
    Create a backbone model using timm library.

    Args:
        backbone_name: Name of the backbone architecture
        pretrained: Whether to use pretrained weights
        num_classes: Number of output classes (for initial creation, will be replaced)

    Returns:
        Tuple of (model, feature_dim) where feature_dim is the size of the feature vector
    """
    # Create model using timm
    model = timm.create_model(
        backbone_name,
        pretrained=pretrained,
        num_classes=0,  # Remove classification head to get features
    )

    # Get feature dimension
    if hasattr(model, "num_features"):
        feature_dim = model.num_features
    elif hasattr(model, "feature_dim"):
        feature_dim = model.feature_dim
    else:
        # Default fallback - create a dummy input to infer
        import torch

        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = model(dummy_input)
            feature_dim = features.shape[-1]

    return model, feature_dim


def create_resnet50(pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Create ResNet50 backbone."""
    return create_backbone("resnet50", pretrained=pretrained)


def create_vit_base(pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Create Vision Transformer Base backbone."""
    return create_backbone("vit_base_patch16_224", pretrained=pretrained)


def create_efficientnet_b0(pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Create EfficientNet-B0 backbone."""
    return create_backbone("efficientnet_b0", pretrained=pretrained)
