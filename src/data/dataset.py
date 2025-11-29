"""Dataset class for emotion classification."""

from pathlib import Path
from typing import Callable, Optional, Tuple, List

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class EmotionDataset(Dataset):
    """
    Dataset for emotion classification from images.

    Supports both single-label and multi-label classification.
    """

    def __init__(
        self,
        image_paths: List[Path],
        labels: List[np.ndarray],
        transform: Optional[Callable] = None,
        multi_label: bool = True,
    ):
        """
        Initialize the emotion dataset.

        Args:
            image_paths: List of paths to images
            labels: List of label arrays (multi-hot for multi-label, or one-hot for single-label)
            transform: Optional transform to apply to images
            multi_label: Whether this is multi-label classification
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.multi_label = multi_label

        # Validate inputs
        assert len(image_paths) == len(labels), "Number of images and labels must match"

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (image_tensor, label_tensor)
        """
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label

    @classmethod
    def from_directory(
        cls,
        data_dir: Path,
        annotation_file: Path,
        transform: Optional[Callable] = None,
        multi_label: bool = True,
    ) -> "EmotionDataset":
        """
        Create dataset from a directory and annotation file.

        Args:
            data_dir: Directory containing images
            annotation_file: Path to annotation file (CSV or JSON)
            transform: Optional transform to apply
            multi_label: Whether this is multi-label classification

        Returns:
            EmotionDataset instance
        """
        # This will be implemented based on your specific dataset format
        # Placeholder for now
        raise NotImplementedError("Implement based on your dataset format")


class AffectNetDataset(EmotionDataset):
    """Dataset class specifically for AffectNet dataset."""

    AFFECTNET_EMOTIONS = [
        "neutral",
        "happy",
        "sad",
        "surprise",
        "fear",
        "disgust",
        "angry",
        "contempt",
    ]

    # Map AffectNet labels to our 8 emotion categories
    AFFECTNET_TO_EMOTION_MAP = {
        0: 6,  # neutral -> neutral
        1: 0,  # happy -> happy
        2: 1,  # sad -> sad
        3: 4,  # surprise -> surprise
        4: 3,  # fear -> fear
        5: 5,  # disgust -> disgust
        6: 2,  # angry -> angry
        7: 7,  # contempt -> other
    }

    @classmethod
    def map_label(cls, affectnet_label: int) -> int:
        """Map AffectNet label to our emotion label."""
        return cls.AFFECTNET_TO_EMOTION_MAP.get(affectnet_label, 7)


class FIDataset(EmotionDataset):
    """Dataset class specifically for Flickr & Instagram (FI) dataset."""

    FI_EMOTIONS = [
        "amusement",
        "anger",
        "awe",
        "contentment",
        "disgust",
        "excitement",
        "fear",
        "sadness",
    ]

    # Map FI labels to our 8 emotion categories
    FI_TO_EMOTION_MAP = {
        "amusement": 0,  # -> happy
        "anger": 2,  # -> angry
        "awe": 4,  # -> surprise
        "contentment": 0,  # -> happy
        "disgust": 5,  # -> disgust
        "excitement": 0,  # -> happy
        "fear": 3,  # -> fear
        "sadness": 1,  # -> sad
    }

    @classmethod
    def map_label(cls, fi_label: str) -> int:
        """Map FI label to our emotion label."""
        return cls.FI_TO_EMOTION_MAP.get(fi_label, 7)
