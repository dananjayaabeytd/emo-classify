"""Main entry point for the emotion classification system."""

import argparse
from pathlib import Path


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Emotion Classification System for Social Media Images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  python -m scripts.train --data-dir ./data/train --output-dir ./models
  
  # Run inference on an image
  python -m scripts.inference --checkpoint ./models/best_model.pth --image ./test_image.jpg
  
  # Start the API server
  python -m scripts.run_api --checkpoint ./models/best_model.pth --port 8000
        """,
    )
    
    parser.add_argument(
        "command",
        choices=["train", "inference", "api", "info"],
        help="Command to run",
    )
    
    args = parser.parse_args()
    
    if args.command == "info":
        print_info()
    elif args.command == "train":
        print("Use: python -m scripts.train --help for training options")
    elif args.command == "inference":
        print("Use: python -m scripts.inference --help for inference options")
    elif args.command == "api":
        print("Use: python -m scripts.run_api --help for API server options")


def print_info():
    """Print system information."""
    print("=" * 70)
    print("Emotion Classification System for Social Media Images")
    print("=" * 70)
    print("\nSupported Emotions:")
    from src.config.emotion_config import EmotionConfig
    for i, emotion in enumerate(EmotionConfig.EMOTIONS, 1):
        emojis = " ".join(EmotionConfig.EMOTION_TO_EMOJIS[emotion][:5])
        print(f"  {i}. {emotion.capitalize()}: {emojis} ...")
    
    print("\nProject Structure:")
    print("  src/")
    print("    ├── config/      - Configuration files")
    print("    ├── data/        - Data loading and preprocessing")
    print("    ├── models/      - Model architecture")
    print("    ├── training/    - Training loop and metrics")
    print("    ├── inference/   - Inference and prediction")
    print("    └── api/         - FastAPI application")
    print("  scripts/           - Training and inference scripts")
    print("  data/              - Dataset storage")
    print("  models/            - Trained model checkpoints")
    
    print("\nAvailable Commands:")
    print("  train      - Train a new model")
    print("  inference  - Run inference on images")
    print("  api        - Start the API server")
    print("  info       - Show this information")
    print("=" * 70)


if __name__ == "__main__":
    main()
