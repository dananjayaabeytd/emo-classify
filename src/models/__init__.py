"""Model architecture modules."""

from .emotion_classifier import EmotionClassifier
from .backbone import create_backbone

__all__ = ["EmotionClassifier", "create_backbone"]
