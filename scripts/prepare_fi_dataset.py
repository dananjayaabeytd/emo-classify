"""Script to download and prepare the FI dataset."""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.fi_dataset import FIDataset, create_fi_splits


def main():
    parser = argparse.ArgumentParser(
        description="Prepare FI (Flickr & Instagram) Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=FIDataset.download_instructions(),
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show dataset information")

    # Create splits command
    split_parser = subparsers.add_parser("create-splits", help="Create train/val/test splits")
    split_parser.add_argument(
        "--annotation-file",
        type=str,
        required=True,
        help="Path to full annotation file",
    )
    split_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for split files",
    )
    split_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)",
    )
    split_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)",
    )
    split_parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test set ratio (default: 0.15)",
    )
    split_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify dataset structure")
    verify_parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory of FI dataset",
    )
    verify_parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Which split to verify",
    )

    args = parser.parse_args()

    if args.command == "info":
        print("=" * 70)
        print("FI (Flickr & Instagram) Dataset Information")
        print("=" * 70)
        print()
        print("Paper: Building a large scale dataset for image emotion recognition:")
        print("       The fine print and the benchmark")
        print("Authors: You Q, Luo J, Jin H, et al.")
        print("Conference: AAAI 2016")
        print()
        print("Dataset Statistics:")
        print("  - Total images: 23,308 (with at least 3 AMT votes)")
        print("  - Original size: 90,000 (noisy, before filtering)")
        print("  - Source: Flickr and Instagram")
        print("  - Emotions: 8 categories (Mikels' model)")
        print()
        print("Emotion Categories:")
        for i, emotion in enumerate(FIDataset.FI_EMOTIONS, 1):
            our_emotion = FIDataset.FI_TO_EMOTION_MAP[emotion]
            print(f"  {i}. {emotion.capitalize()} → {our_emotion.capitalize()}")
        print()
        print(FIDataset.download_instructions())

    elif args.command == "create-splits":
        print("Creating train/val/test splits...")
        create_fi_splits(
            annotation_file=Path(args.annotation_file),
            output_dir=Path(args.output_dir),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        print("\n✅ Splits created successfully!")

    elif args.command == "verify":
        print(f"Verifying FI dataset: {args.split} split...")
        try:
            from src.config.model_config import ModelConfig
            from src.data.transforms import get_val_transforms

            config = ModelConfig()
            transform = get_val_transforms(config)

            dataset = FIDataset.from_directory(
                Path(args.data_dir),
                split=args.split,
                transform=transform,
            )

            print(f"\n✅ Dataset loaded successfully!")
            print(f"   Total samples: {len(dataset)}")

            # Test loading a sample
            if len(dataset) > 0:
                print("\nTesting sample loading...")
                image, label = dataset[0]
                print(f"   Image shape: {image.shape}")
                print(f"   Label shape: {label.shape}")
                print(f"   Label values: {label}")

                # Find which emotions are active
                from src.config.emotion_config import EmotionConfig

                active_emotions = [
                    EmotionConfig.IDX_TO_EMOTION[i]
                    for i, val in enumerate(label)
                    if val > 0
                ]
                print(f"   Active emotions: {active_emotions}")

                print("\n✅ Dataset verification passed!")

        except Exception as e:
            print(f"\n❌ Dataset verification failed: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
