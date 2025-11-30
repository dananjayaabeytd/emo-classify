"""
Automated FER2013 Dataset Downloader

This script downloads and extracts FER2013 dataset from Kaggle.
Requires Kaggle API credentials.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import zipfile
import shutil


def check_kaggle_installed() -> bool:
    """Check if Kaggle CLI is installed."""
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def install_kaggle():
    """Install Kaggle CLI."""
    print("📦 Installing Kaggle CLI...")
    try:
        subprocess.run(
            ["uv", "pip", "install", "kaggle"],
            check=True
        )
        print("✅ Kaggle CLI installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Kaggle: {e}")
        return False


def check_kaggle_credentials() -> bool:
    """Check if Kaggle API credentials are configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def show_credential_instructions():
    """Show instructions for setting up Kaggle credentials."""
    print("\n" + "=" * 70)
    print("Kaggle API Credentials Required")
    print("=" * 70)
    print("\n📝 Step 1: Get Your API Key")
    print("   1. Go to: https://www.kaggle.com/")
    print("   2. Sign in to your account")
    print("   3. Click on your profile picture → 'Settings'")
    print("   4. Scroll to 'API' section")
    print("   5. Click 'Create New Token'")
    print("   6. Download kaggle.json file")
    
    print("\n📁 Step 2: Install API Key")
    kaggle_dir = Path.home() / ".kaggle"
    print(f"   Move kaggle.json to: {kaggle_dir}")
    print(f"   Full path: {kaggle_dir / 'kaggle.json'}")
    
    print("\n💻 Quick Setup Commands (PowerShell):")
    print(f"   mkdir {kaggle_dir}")
    print(f"   move Downloads\\kaggle.json {kaggle_dir}\\")
    
    print("\n✅ Step 3: Verify Setup")
    print("   Run: kaggle datasets list")
    print("   You should see a list of datasets")
    
    print("\n" + "=" * 70 + "\n")


def download_fer2013(output_dir: Path) -> bool:
    """
    Download FER2013 dataset from Kaggle.
    
    Args:
        output_dir: Directory to save the dataset
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n📥 Downloading FER2013 dataset to: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download dataset
        print("⏳ Downloading... (this may take a few minutes)")
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                "msambare/fer2013",
                "-p",
                str(output_dir),
            ],
            check=True,
        )
        
        print("✅ Download completed!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        return False


def extract_dataset(zip_path: Path, extract_to: Path) -> bool:
    """
    Extract the downloaded dataset.
    
    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract files to
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n📦 Extracting dataset...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        print(f"✅ Extraction completed!")
        
        # Remove zip file
        zip_path.unlink()
        print(f"🗑️  Removed zip file: {zip_path.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False


def verify_structure(data_dir: Path) -> bool:
    """Verify the extracted dataset structure."""
    print(f"\n🔍 Verifying dataset structure...")
    
    required_dirs = [
        data_dir / "train",
        data_dir / "test",
    ]
    
    emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    
    for split_dir in required_dirs:
        if not split_dir.exists():
            print(f"❌ Missing directory: {split_dir}")
            return False
        
        for emotion in emotions:
            emotion_dir = split_dir / emotion
            if not emotion_dir.exists():
                print(f"❌ Missing emotion directory: {emotion_dir}")
                return False
    
    print("✅ Dataset structure verified!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download FER2013 dataset from Kaggle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/fer2013"),
        help="Output directory for the dataset (default: data/fer2013)"
    )
    
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip dataset structure verification"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("FER2013 Dataset Downloader")
    print("=" * 70)
    
    # Check Kaggle CLI
    if not check_kaggle_installed():
        print("\n❌ Kaggle CLI not found!")
        response = input("\n📦 Install Kaggle CLI now? (y/n): ")
        if response.lower() == 'y':
            if not install_kaggle():
                sys.exit(1)
        else:
            print("\n💡 Install manually: uv pip install kaggle")
            sys.exit(1)
    
    print("✅ Kaggle CLI found")
    
    # Check credentials
    if not check_kaggle_credentials():
        print("\n❌ Kaggle API credentials not found!")
        show_credential_instructions()
        sys.exit(1)
    
    print("✅ Kaggle credentials found")
    
    # Download dataset
    if not download_fer2013(args.output_dir):
        sys.exit(1)
    
    # Find and extract zip file
    zip_files = list(args.output_dir.glob("*.zip"))
    if not zip_files:
        print("❌ No zip file found after download!")
        sys.exit(1)
    
    zip_path = zip_files[0]
    if not extract_dataset(zip_path, args.output_dir):
        sys.exit(1)
    
    # Verify structure
    if not args.skip_verification:
        if not verify_structure(args.output_dir):
            print("\n⚠️  Dataset structure verification failed!")
            print("Please check the extracted files manually.")
            sys.exit(1)
    
    print("\n" + "=" * 70)
    print("✅ FER2013 Dataset Ready!")
    print("=" * 70)
    print(f"\nDataset location: {args.output_dir.absolute()}")
    print("\n🚀 Next Steps:")
    print(f"1. Verify: python -m scripts.prepare_fer2013 verify --data-dir {args.output_dir}")
    print(f"2. Train:  python -m scripts.train --data-dir {args.output_dir} --dataset-type fer2013 --epochs 50")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
