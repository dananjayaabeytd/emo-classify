"""Inference script for testing emotion classification."""

import argparse
from pathlib import Path
from PIL import Image

from src.inference.predictor import EmotionPredictor


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Run emotion classification inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--threshold", type=float, default=0.35, help="Classification threshold")
    parser.add_argument("--show-scores", action="store_true", help="Show emotion scores")
    
    args = parser.parse_args()
    
    # Load predictor
    print(f"Loading model from {args.checkpoint}...")
    predictor = EmotionPredictor.from_checkpoint(
        Path(args.checkpoint),
        device=args.device,
        threshold=args.threshold,
    )
    
    # Load and predict
    print(f"Processing image: {args.image}")
    result = predictor.predict(args.image, return_probabilities=args.show_scores)
    
    # Display results
    print("\n" + "=" * 50)
    print("EMOTION CLASSIFICATION RESULTS")
    print("=" * 50)
    print(f"\nPredicted Emotions: {', '.join(result['predicted_emotions'])}")
    print(f"\nAllowed Emojis: {' '.join(result['allowed_emojis'])}")
    
    if args.show_scores and "emotion_scores" in result:
        print("\nEmotion Scores:")
        for emotion, score in result["emotion_scores"].items():
            print(f"  {emotion}: {score:.4f}")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
