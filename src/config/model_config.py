"""Model architecture configuration."""

from dataclasses import dataclass, field
from typing import Literal, List


@dataclass
class ModelConfig:
    """Configuration for the emotion classification model."""

    # Model architecture
    backbone: Literal["vit_base_patch16_224", "resnet50", "efficientnet_b0"] = "vit_base_patch16_224"
    pretrained: bool = True
    num_classes: int = 8  # 8 emotions
    
    # Emotion labels
    emotions: List[str] = field(default_factory=lambda: [
        "happy", "sad", "angry", "fear", 
        "surprise", "disgust", "neutral", "other"
    ])

    # Image preprocessing
    image_size: int = 224
    mean: tuple = (0.485, 0.456, 0.406)  # ImageNet mean
    std: tuple = (0.229, 0.224, 0.225)  # ImageNet std

    # Model output
    multi_label: bool = True  # Multi-label classification (BCEWithLogitsLoss)

    # Dropout
    dropout: float = 0.1

    def __post_init__(self):
        """Validate configuration."""
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
