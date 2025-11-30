"""Emotion prediction from images."""

import torch
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Union
import numpy as np

from ..models.emotion_classifier import EmotionClassifier
from ..config.emotion_config import EmotionConfig
from ..config.model_config import ModelConfig
from ..data.transforms import get_inference_transforms


class EmotionPredictor:
    """Predictor class for emotion classification inference."""

    def __init__(
        self,
        model: EmotionClassifier,
        model_config: ModelConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        threshold: float = 0.35,
    ):
        """
        Initialize the emotion predictor.

        Args:
            model: Trained emotion classification model
            model_config: Model configuration
            device: Device to run inference on
            threshold: Threshold for multi-label classification
        """
        self.model = model.to(device)
        self.model.eval()
        self.model_config = model_config
        self.device = device
        self.threshold = threshold

        # Get transforms
        self.transform = get_inference_transforms(model_config)

        # Emotion config
        self.emotion_config = EmotionConfig()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        threshold: float = 0.35,
    ) -> "EmotionPredictor":
        """
        Create predictor from a saved checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
            threshold: Threshold for multi-label classification

        Returns:
            EmotionPredictor instance
        """
        # Load checkpoint (with weights_only=False for PyTorch 2.6+)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_config = checkpoint["model_config"]

        # Create model
        model = EmotionClassifier(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])

        return cls(model, model_config, device, threshold)

    def preprocess_image(self, image: Union[Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess an image for inference.

        Args:
            image: Image as path, PIL Image, or numpy array

        Returns:
            Preprocessed image tensor
        """
        # Convert to PIL Image if needed
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            # Ensure PIL Image is RGB
            image = image.convert("RGB")

        # Apply transforms
        image_tensor = self.transform(image)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor

    @torch.no_grad()
    def predict(
        self,
        image: Union[Path, Image.Image, np.ndarray],
        return_probabilities: bool = False,
    ) -> Dict[str, any]:
        """
        Predict emotions from an image.

        Args:
            image: Input image
            return_probabilities: Whether to return probabilities for all emotions

        Returns:
            Dictionary containing predictions and allowed emojis
        """
        # Preprocess
        image_tensor = self.preprocess_image(image).to(self.device)

        # Predict
        predictions, probabilities = self.model.predict_emotions(
            image_tensor, threshold=self.threshold
        )

        # Convert to numpy
        predictions = predictions.cpu().numpy()[0]
        probabilities = probabilities.cpu().numpy()[0]

        # Get predicted emotion labels
        predicted_indices = np.where(predictions > 0)[0]
        predicted_emotions = [
            self.emotion_config.IDX_TO_EMOTION[idx] for idx in predicted_indices
        ]

        # Get allowed emojis
        allowed_emojis = self.emotion_config.get_allowed_emojis(predicted_emotions)

        # Build result
        result = {
            "predicted_emotions": predicted_emotions,
            "allowed_emojis": allowed_emojis,
            "emotion_scores": {
                self.emotion_config.IDX_TO_EMOTION[i]: float(prob)
                for i, prob in enumerate(probabilities)
            },
        }

        if not return_probabilities:
            del result["emotion_scores"]

        return result

    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[Path, Image.Image, np.ndarray]],
        return_probabilities: bool = False,
    ) -> List[Dict[str, any]]:
        """
        Predict emotions for a batch of images.

        Args:
            images: List of input images
            return_probabilities: Whether to return probabilities

        Returns:
            List of prediction dictionaries
        """
        results = []
        for image in images:
            result = self.predict(image, return_probabilities)
            results.append(result)

        return results

    def predict_with_visualization(
        self,
        image: Union[Path, Image.Image, np.ndarray],
    ) -> Tuple[Dict[str, any], Image.Image]:
        """
        Predict emotions and create a visualization.

        Args:
            image: Input image

        Returns:
            Tuple of (prediction_dict, annotated_image)
        """
        # Get predictions
        result = self.predict(image, return_probabilities=True)

        # Load original image if path
        if isinstance(image, (str, Path)):
            original_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            original_image = Image.fromarray(image).convert("RGB")
        else:
            original_image = image

        # Create visualization (simple version - can be enhanced)
        # For now, just return the original image
        # You can add text annotations, emoji overlays, etc.

        return result, original_image

    def get_allowed_emojis_only(
        self,
        image: Union[Path, Image.Image, np.ndarray],
    ) -> List[str]:
        """
        Get only the allowed emojis for an image (simplified interface).

        Args:
            image: Input image

        Returns:
            List of allowed emoji strings
        """
        result = self.predict(image, return_probabilities=False)
        return result["allowed_emojis"]
