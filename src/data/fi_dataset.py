"""
FI (Flickr & Instagram) Dataset Loader

Dataset Information:
- Paper: "Building a large scale dataset for image emotion recognition: The fine print and the benchmark"
  You Q, Luo J, Jin H, et al. AAAI 2016.
- Size: 23,308 images (with at least 3 votes from AMT workers)
- Source: Flickr and Instagram
- Emotions: 8 categories based on Mikels' wheel of emotions
- Download: https://qzyou.github.io/ or http://47.105.62.179:8081/sentiment/index.html

Emotion Categories (Mikels' Model):
1. Amusement
2. Anger  
3. Awe
4. Contentment
5. Disgust
6. Excitement
7. Fear
8. Sadness

Note: These map to our 8 emotions as follows:
- Amusement → Happy
- Anger → Angry
- Awe → Surprise
- Contentment → Happy
- Disgust → Disgust
- Excitement → Happy
- Fear → Fear
- Sadness → Sad
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from PIL import Image
import json

from .dataset import EmotionDataset
from ..config.emotion_config import EmotionConfig


class FIDataset(EmotionDataset):
    """
    FI (Flickr & Instagram) Dataset Loader.
    
    Expected dataset structure:
    fi_dataset/
    ├── images/
    │   ├── flickr/
    │   │   ├── 123456.jpg
    │   │   └── ...
    │   └── instagram/
    │       ├── 789012.jpg
    │       └── ...
    └── annotations/
        ├── train.txt (or train.csv)
        ├── val.txt
        └── test.txt
    
    Annotation format (space or comma separated):
    image_name emotion_label
    or
    image_name,emotion_label
    
    Where emotion_label is one of: amusement, anger, awe, contentment, disgust, excitement, fear, sadness
    """

    # FI dataset emotions (Mikels' model)
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

    # Map FI emotions to our 8 emotion categories
    FI_TO_EMOTION_MAP = {
        "amusement": "happy",
        "anger": "angry",
        "awe": "surprise",
        "contentment": "happy",
        "disgust": "disgust",
        "excitement": "happy",
        "fear": "fear",
        "sadness": "sad",
    }

    def __init__(
        self,
        image_paths: List[Path],
        labels: List[np.ndarray],
        transform=None,
        multi_label: bool = True,
    ):
        """Initialize FI dataset."""
        super().__init__(image_paths, labels, transform, multi_label)

    @classmethod
    def from_annotation_file(
        cls,
        annotation_file: Path,
        image_dir: Path,
        transform=None,
        multi_label: bool = True,
    ) -> "FIDataset":
        """
        Load FI dataset from annotation file.

        Args:
            annotation_file: Path to annotation file (txt or csv)
            image_dir: Directory containing images
            transform: Image transforms
            multi_label: Whether to use multi-label format

        Returns:
            FIDataset instance
        """
        print(f"Loading FI dataset from {annotation_file}...")

        image_paths = []
        emotion_labels = []

        # Read annotation file
        with open(annotation_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Parse line (handle both space and comma separated)
                if "," in line:
                    parts = line.split(",")
                else:
                    parts = line.split()

                if len(parts) < 2:
                    continue

                image_name = parts[0]
                fi_emotion = parts[1].lower().strip()

                # Find image path (check both flickr and instagram subdirs)
                image_path = None
                for subdir in ["flickr", "instagram", "."]:
                    candidate = image_dir / subdir / image_name
                    if candidate.exists():
                        image_path = candidate
                        break

                if image_path is None:
                    # Try without subdirectory
                    candidate = image_dir / image_name
                    if candidate.exists():
                        image_path = candidate
                    else:
                        print(f"Warning: Image not found: {image_name}")
                        continue

                # Map FI emotion to our emotion category
                our_emotion = cls.FI_TO_EMOTION_MAP.get(fi_emotion)
                if our_emotion is None:
                    print(f"Warning: Unknown FI emotion: {fi_emotion}")
                    continue

                image_paths.append(image_path)
                emotion_labels.append(our_emotion)

        # Convert to multi-hot labels
        labels = cls._convert_to_multihot(emotion_labels)

        print(f"Loaded {len(image_paths)} images from FI dataset")
        print(f"Emotion distribution:")
        emotion_counts = cls._count_emotions(emotion_labels)
        for emotion, count in emotion_counts.items():
            print(f"  {emotion}: {count}")

        return cls(image_paths, labels, transform, multi_label)

    @classmethod
    def from_csv(
        cls,
        csv_file: Path,
        image_dir: Path,
        transform=None,
        multi_label: bool = True,
    ) -> "FIDataset":
        """
        Load FI dataset from CSV file.

        Expected CSV format:
        image_name,emotion
        123456.jpg,amusement
        789012.jpg,fear

        Args:
            csv_file: Path to CSV file
            image_dir: Directory containing images
            transform: Image transforms
            multi_label: Whether to use multi-label format

        Returns:
            FIDataset instance
        """
        print(f"Loading FI dataset from CSV: {csv_file}...")

        df = pd.read_csv(csv_file)

        if "image_name" not in df.columns or "emotion" not in df.columns:
            raise ValueError("CSV must have 'image_name' and 'emotion' columns")

        image_paths = []
        emotion_labels = []

        for _, row in df.iterrows():
            image_name = row["image_name"]
            fi_emotion = str(row["emotion"]).lower().strip()

            # Find image path
            image_path = None
            for subdir in ["flickr", "instagram", "."]:
                candidate = image_dir / subdir / image_name
                if candidate.exists():
                    image_path = candidate
                    break

            if image_path is None:
                candidate = image_dir / image_name
                if candidate.exists():
                    image_path = candidate
                else:
                    continue

            # Map emotion
            our_emotion = cls.FI_TO_EMOTION_MAP.get(fi_emotion)
            if our_emotion is None:
                continue

            image_paths.append(image_path)
            emotion_labels.append(our_emotion)

        # Convert to multi-hot labels
        labels = cls._convert_to_multihot(emotion_labels)

        print(f"Loaded {len(image_paths)} images from FI dataset")

        return cls(image_paths, labels, transform, multi_label)

    @classmethod
    def from_directory(
        cls,
        data_dir: Path,
        split: str = "train",
        transform=None,
        multi_label: bool = True,
    ) -> "FIDataset":
        """
        Load FI dataset from standard directory structure.

        Expected structure:
        data_dir/
        ├── images/
        └── annotations/
            ├── train.txt
            ├── val.txt
            └── test.txt

        Args:
            data_dir: Root directory of FI dataset
            split: One of 'train', 'val', 'test'
            transform: Image transforms
            multi_label: Whether to use multi-label format

        Returns:
            FIDataset instance
        """
        annotation_file = data_dir / "annotations" / f"{split}.txt"
        if not annotation_file.exists():
            # Try CSV
            annotation_file = data_dir / "annotations" / f"{split}.csv"
            if annotation_file.exists():
                image_dir = data_dir / "images"
                return cls.from_csv(annotation_file, image_dir, transform, multi_label)

        image_dir = data_dir / "images"
        return cls.from_annotation_file(annotation_file, image_dir, transform, multi_label)

    @staticmethod
    def _convert_to_multihot(emotion_labels: List[str]) -> List[np.ndarray]:
        """Convert emotion labels to multi-hot encoding."""
        labels = []
        for emotion in emotion_labels:
            # Create multi-hot vector
            label = np.zeros(EmotionConfig.NUM_EMOTIONS, dtype=np.float32)
            if emotion in EmotionConfig.EMOTION_TO_IDX:
                label[EmotionConfig.EMOTION_TO_IDX[emotion]] = 1.0
            labels.append(label)
        return labels

    @staticmethod
    def _count_emotions(emotion_labels: List[str]) -> Dict[str, int]:
        """Count emotion occurrences."""
        counts = {}
        for emotion in emotion_labels:
            counts[emotion] = counts.get(emotion, 0) + 1
        return counts

    @classmethod
    def download_instructions(cls) -> str:
        """Return instructions for downloading the FI dataset."""
        return """
FI (Flickr & Instagram) Dataset Download Instructions:

1. Visit the official sources:
   - Author's homepage: https://qzyou.github.io/
   - Nankai CV Lab: http://47.105.62.179:8081/sentiment/index.html

2. Request access to the dataset (you may need to fill out a form)

3. Once downloaded, organize the dataset as:
   fi_dataset/
   ├── images/
   │   ├── flickr/
   │   └── instagram/
   └── annotations/
       ├── train.txt
       ├── val.txt
       └── test.txt

4. Annotation format (one line per image):
   image_name emotion_label
   
   Example:
   123456.jpg amusement
   789012.jpg fear
   
   OR in CSV format:
   image_name,emotion
   123456.jpg,amusement
   789012.jpg,fear

5. Place the dataset in your data directory and use:
   dataset = FIDataset.from_directory(Path("data/fi_dataset"), split="train")

For questions, refer to the paper:
You Q, Luo J, Jin H, et al. "Building a large scale dataset for image emotion 
recognition: The fine print and the benchmark." AAAI 2016.
"""


def create_fi_splits(
    annotation_file: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    shuffle: bool = True,
    seed: int = 42,
):
    """
    Create train/val/test splits from a single annotation file.

    Args:
        annotation_file: Path to full annotation file
        output_dir: Directory to save split files
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        shuffle: Whether to shuffle before splitting
        seed: Random seed for reproducibility
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read all annotations
    with open(annotation_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Shuffle if requested
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(lines)

    # Calculate split indices
    total = len(lines)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    # Split
    train_lines = lines[:train_end]
    val_lines = lines[train_end:val_end]
    test_lines = lines[val_end:]

    # Save splits
    for split_name, split_lines in [
        ("train", train_lines),
        ("val", val_lines),
        ("test", test_lines),
    ]:
        output_file = output_dir / f"{split_name}.txt"
        with open(output_file, "w") as f:
            f.write("\n".join(split_lines))
        print(f"Saved {len(split_lines)} samples to {output_file}")

    print(f"\nSplit summary:")
    print(f"  Train: {len(train_lines)} ({len(train_lines)/total*100:.1f}%)")
    print(f"  Val:   {len(val_lines)} ({len(val_lines)/total*100:.1f}%)")
    print(f"  Test:  {len(test_lines)} ({len(test_lines)/total*100:.1f}%)")
