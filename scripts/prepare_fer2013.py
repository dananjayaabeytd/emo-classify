"""
FER2013 Dataset Preparation Script

This script helps you download and prepare the FER2013 dataset.
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.fer2013_dataset import FER2013Dataset, print_fer2013_info


def verify_dataset(data_dir: Path, split: str = "train"):
    """Verify FER2013 dataset structure."""
    print(f"\n{'=' * 60}")
    print(f"Verifying FER2013 Dataset: {split} split")
    print(f"{'=' * 60}\n")
    
    try:
        # Try to load dataset
        dataset = FER2013Dataset.from_directory(
            data_dir=data_dir,
            split=split,
        )
        
        print(f"\n✅ Dataset verification successful!")
        print(f"✅ Found {len(dataset)} images in {split} split")
        
        # Test loading a sample
        if len(dataset) > 0:
            print(f"\n🧪 Testing data loading...")
            image, label = dataset[0]
            print(f"✅ Successfully loaded sample image")
            print(f"   - Image type: {type(image)}")
            print(f"   - Label shape: {label.shape}")
            print(f"   - Label: {label}")
        
        print(f"\n✅ All checks passed! Dataset is ready for training.")
        return True
        
    except Exception as e:
        print(f"\n❌ Dataset verification failed!")
        print(f"Error: {e}")
        print(f"\nPlease check:")
        print(f"  1. Dataset is downloaded from Kaggle")
        print(f"  2. Files are extracted to: {data_dir}")
        print(f"  3. Directory structure is correct")
        return False


def show_download_instructions():
    """Show download instructions for FER2013."""
    print("\n" + "=" * 70)
    print("FER2013 Dataset Download Instructions")
    print("=" * 70)
    print("\n📥 Step 1: Create Kaggle Account")
    print("   Go to: https://www.kaggle.com/account/login")
    print("   Sign up for a free account (if you don't have one)")
    
    print("\n📥 Step 2: Download Dataset")
    print("   Go to: https://www.kaggle.com/datasets/msambare/fer2013")
    print("   Click the 'Download' button")
    print("   File: archive.zip (~60 MB)")
    
    print("\n📦 Step 3: Extract Files")
    print("   Extract archive.zip to: data/fer2013/")
    print("   Expected structure:")
    print("   data/fer2013/")
    print("   ├── train/")
    print("   │   ├── angry/")
    print("   │   ├── disgust/")
    print("   │   ├── fear/")
    print("   │   ├── happy/")
    print("   │   ├── sad/")
    print("   │   ├── surprise/")
    print("   │   └── neutral/")
    print("   └── test/")
    print("       └── [same structure]")
    
    print("\n✅ Step 4: Verify Dataset")
    print("   Run: python -m scripts.prepare_fer2013 verify --data-dir data/fer2013")
    
    print("\n🚀 Step 5: Start Training")
    print("   Run: python -m scripts.train --data-dir data/fer2013 --dataset-type fer2013 --epochs 50")
    
    print("\n" + "=" * 70)
    print("Alternative: Download via Kaggle CLI")
    print("=" * 70)
    print("1. Install: pip install kaggle")
    print("2. Setup API key: https://www.kaggle.com/docs/api")
    print("3. Download: kaggle datasets download -d msambare/fer2013")
    print("4. Extract: unzip fer2013.zip -d data/fer2013/")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="FER2013 Dataset Preparation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Info command
    subparsers.add_parser(
        "info",
        help="Show FER2013 dataset information"
    )
    
    # Download command
    subparsers.add_parser(
        "download",
        help="Show download instructions for FER2013"
    )
    
    # Verify command
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify FER2013 dataset structure"
    )
    verify_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/fer2013"),
        help="Path to FER2013 dataset directory"
    )
    verify_parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "validation"],
        help="Dataset split to verify"
    )
    
    args = parser.parse_args()
    
    if args.command == "info":
        print_fer2013_info()
    
    elif args.command == "download":
        show_download_instructions()
    
    elif args.command == "verify":
        verify_dataset(args.data_dir, args.split)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
