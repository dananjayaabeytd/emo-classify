"""
FER2013 Dataset Loader

FER2013 is a publicly available emotion recognition dataset with 48,000 images.
Download from: https://www.kaggle.com/datasets/msambare/fer2013
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, List
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class FER2013Dataset(Dataset):
    """
    FER2013 Dataset Loader
    
    FER2013 contains 48,000 48x48 grayscale face images with 7 emotion categories.
    
    Emotion Mapping:
    - FER2013 emotions: angry, disgust, fear, happy, sad, surprise, neutral
    - System emotions: angry, disgust, fear, happy, sad, surprise, neutral, other
    
    Dataset Statistics:
    - Training: 28,709 images
    - Validation: 3,589 images
    - Testing: 3,589 images
    
    Download from: https://www.kaggle.com/datasets/msambare/fer2013
    
    Expected directory structure:
    data/fer2013/
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── sad/
    │   ├── surprise/
    │   └── neutral/
    ├── test/
    │   └── [same structure]
    └── validation/ (optional)
        └── [same structure]
    """
    
    # FER2013 emotion labels
    FER_EMOTIONS = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "sad",
        "surprise",
        "neutral",
    ]
    
    # Mapping from FER2013 to your system's 8 emotions
    FER_TO_SYSTEM_EMOTION = {
        "angry": "angry",
        "disgust": "disgust",
        "fear": "fear",
        "happy": "happy",
        "sad": "sad",
        "surprise": "surprise",
        "neutral": "neutral",
    }
    
    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        transform=None,
        system_emotions: Optional[List[str]] = None,
    ):
        """
        Initialize FER2013 dataset.
        
        Args:
            data_dir: Path to FER2013 dataset directory
            split: Dataset split ("train", "test", or "validation")
            transform: Image transforms to apply
            system_emotions: List of system emotion labels (8 emotions)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Default system emotions
        if system_emotions is None:
            system_emotions = [
                "happy", "sad", "angry", "fear", 
                "surprise", "disgust", "neutral", "other"
            ]
        self.system_emotions = system_emotions
        
        # Load image paths and labels
        self.samples = self._load_samples()
        
        print(f"Loaded FER2013 {split} set: {len(self.samples)} images")
        self._print_emotion_distribution()
    
    def _load_samples(self) -> List[Tuple[Path, str]]:
        """Load all image paths and their emotion labels."""
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}\n"
                f"Expected structure: {self.data_dir}/{self.split}/[emotion_folders]/"
            )
        
        samples = []
        
        # Iterate through emotion folders
        for emotion in self.FER_EMOTIONS:
            emotion_dir = split_dir / emotion
            
            if not emotion_dir.exists():
                print(f"Warning: Emotion folder not found: {emotion_dir}")
                continue
            
            # Get all images in this emotion folder
            image_files = list(emotion_dir.glob("*.jpg")) + \
                         list(emotion_dir.glob("*.png")) + \
                         list(emotion_dir.glob("*.jpeg"))
            
            for img_path in image_files:
                samples.append((img_path, emotion))
        
        if not samples:
            raise ValueError(
                f"No images found in {split_dir}\n"
                f"Make sure images are organized in emotion subfolders."
            )
        
        return samples
    
    def _print_emotion_distribution(self):
        """Print emotion distribution statistics."""
        emotion_counts = {}
        for _, emotion in self.samples:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print(f"\nEmotion distribution ({self.split}):")
        for emotion in self.FER_EMOTIONS:
            count = emotion_counts.get(emotion, 0)
            percentage = (count / len(self.samples)) * 100 if self.samples else 0
            print(f"  {emotion:12s}: {count:5d} ({percentage:5.2f}%)")
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (image, label) where:
            - image: Transformed image tensor
            - label: Multi-label binary vector for 8 emotions
        """
        img_path, fer_emotion = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image in case of error
            image = Image.new("RGB", (48, 48), color=(128, 128, 128))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert FER emotion to system emotion
        system_emotion = self.FER_TO_SYSTEM_EMOTION.get(fer_emotion, "other")
        
        # Create multi-label vector
        label = self._create_label_vector(system_emotion)
        
        return image, label
    
    def _create_label_vector(self, emotion: str) -> torch.Tensor:
        """
        Create multi-label binary vector for the given emotion.
        
        Args:
            emotion: System emotion label
        
        Returns:
            Binary tensor of shape (num_emotions,)
        """
        label = torch.zeros(len(self.system_emotions), dtype=torch.float32)
        
        if emotion in self.system_emotions:
            idx = self.system_emotions.index(emotion)
            label[idx] = 1.0
        else:
            # Unknown emotion -> "other"
            if "other" in self.system_emotions:
                idx = self.system_emotions.index("other")
                label[idx] = 1.0
        
        return label
    
    @classmethod
    def get_dataset_info(cls) -> Dict:
        """Get FER2013 dataset information."""
        return {
            "name": "FER2013",
            "description": "Facial Expression Recognition 2013 Dataset",
            "total_images": 35887,
            "train_images": 28709,
            "val_images": 3589,
            "test_images": 3589,
            "emotions": cls.FER_EMOTIONS,
            "num_emotions": len(cls.FER_EMOTIONS),
            "image_size": "48x48 grayscale (converted to RGB)",
            "source": "Kaggle",
            "download_url": "https://www.kaggle.com/datasets/msambare/fer2013",
            "license": "Public Domain",
        }
    
    @classmethod
    def from_directory(
        cls,
        data_dir: Path,
        split: str = "train",
        transform=None,
        system_emotions: Optional[List[str]] = None,
    ) -> "FER2013Dataset":
        """
        Create FER2013 dataset from directory.
        
        Args:
            data_dir: Path to FER2013 root directory
            split: Dataset split ("train", "test", or "validation")
            transform: Image transforms
            system_emotions: List of system emotion labels
        
        Returns:
            FER2013Dataset instance
        """
        return cls(
            data_dir=data_dir,
            split=split,
            transform=transform,
            system_emotions=system_emotions,
        )


def print_fer2013_info():
    """Print FER2013 dataset information."""
    info = FER2013Dataset.get_dataset_info()
    
    print("=" * 60)
    print(f"Dataset: {info['name']}")
    print("=" * 60)
    print(f"\nDescription: {info['description']}")
    print(f"\nTotal Images: {info['total_images']:,}")
    print(f"  - Training:   {info['train_images']:,}")
    print(f"  - Validation: {info['val_images']:,}")
    print(f"  - Testing:    {info['test_images']:,}")
    print(f"\nEmotions ({info['num_emotions']}):")
    for i, emotion in enumerate(info['emotions'], 1):
        print(f"  {i}. {emotion}")
    print(f"\nImage Size: {info['image_size']}")
    print(f"Source: {info['source']}")
    print(f"License: {info['license']}")
    print(f"\nDownload URL:")
    print(f"  {info['download_url']}")
    print("\n" + "=" * 60)
    print("Download Instructions:")
    print("=" * 60)
    print("1. Go to: https://www.kaggle.com/datasets/msambare/fer2013")
    print("2. Click 'Download' button (requires Kaggle account)")
    print("3. Extract the zip file to: data/fer2013/")
    print("4. Verify structure with: python -m scripts.prepare_fer2013 verify")
    print("=" * 60)


if __name__ == "__main__":
    print_fer2013_info()
