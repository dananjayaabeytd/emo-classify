"""Quick verification test to ensure setup is correct."""

import sys
from pathlib import Path


def test_imports():
    """Test that all core modules can be imported."""
    print("Testing imports...")
    
    try:
        # Core libraries
        import torch
        import torchvision
        import timm
        from PIL import Image
        import numpy as np
        from sklearn.metrics import f1_score
        from fastapi import FastAPI
        import uvicorn
        
        print("✅ Core libraries imported successfully")
        print(f"   - PyTorch: {torch.__version__}")
        print(f"   - torchvision: {torchvision.__version__}")
        print(f"   - timm: {timm.__version__}")
        
        # Project modules
        from src.config.emotion_config import EmotionConfig
        from src.config.model_config import ModelConfig
        from src.config.training_config import TrainingConfig
        from src.models.emotion_classifier import EmotionClassifier
        from src.data.transforms import get_train_transforms, get_val_transforms
        
        print("✅ Project modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_config():
    """Test configuration classes."""
    print("\nTesting configuration...")
    
    try:
        from src.config.emotion_config import EmotionConfig
        from src.config.model_config import ModelConfig
        from src.config.training_config import TrainingConfig
        
        # Test EmotionConfig
        assert len(EmotionConfig.EMOTIONS) == 8
        assert "happy" in EmotionConfig.EMOTIONS
        assert len(EmotionConfig.EMOTION_TO_EMOJIS) == 8
        
        # Test emoji mapping
        happy_emojis = EmotionConfig.get_allowed_emojis(["happy"])
        assert len(happy_emojis) > 0
        
        print("✅ EmotionConfig working")
        
        # Test ModelConfig
        model_config = ModelConfig()
        assert model_config.num_classes == 8
        assert model_config.image_size == 224
        
        print("✅ ModelConfig working")
        
        # Test TrainingConfig
        training_config = TrainingConfig()
        assert training_config.batch_size > 0
        assert training_config.num_epochs > 0
        
        print("✅ TrainingConfig working")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    
    try:
        import torch
        from src.config.model_config import ModelConfig
        from src.models.emotion_classifier import EmotionClassifier
        
        # Create small model for testing
        config = ModelConfig(backbone="resnet50")
        model = EmotionClassifier(config)
        
        print("✅ Model created successfully")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        assert output.shape == (1, 8)
        
        print("✅ Forward pass working")
        print(f"   - Output shape: {output.shape}")
        
        # Test model info
        info = model.get_model_info()
        print(f"   - Total params: {info['total_params']:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False


def test_transforms():
    """Test image transforms."""
    print("\nTesting image transforms...")
    
    try:
        from PIL import Image
        import numpy as np
        from src.config.model_config import ModelConfig
        from src.data.transforms import get_train_transforms, get_val_transforms
        
        config = ModelConfig()
        train_transform = get_train_transforms(config)
        val_transform = get_val_transforms(config)
        
        # Create dummy image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        
        # Test transforms
        train_tensor = train_transform(dummy_image)
        val_tensor = val_transform(dummy_image)
        
        assert train_tensor.shape == (3, 224, 224)
        assert val_tensor.shape == (3, 224, 224)
        
        print("✅ Image transforms working")
        print(f"   - Train output shape: {train_tensor.shape}")
        print(f"   - Val output shape: {val_tensor.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Transform error: {e}")
        return False


def test_api_creation():
    """Test API app creation."""
    print("\nTesting API creation...")
    
    try:
        from pathlib import Path
        from src.api.app import create_app
        
        # Note: This will fail without a model checkpoint, but we can test the function exists
        print("✅ API module importable")
        print("   - Note: Full API test requires a trained model checkpoint")
        
        return True
        
    except Exception as e:
        print(f"❌ API creation error: {e}")
        return False


def run_all_tests():
    """Run all verification tests."""
    print("=" * 70)
    print("Emotion Classification System - Setup Verification")
    print("=" * 70)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Model Creation", test_model_creation()))
    results.append(("Image Transforms", test_transforms()))
    results.append(("API Creation", test_api_creation()))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary:")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Download a dataset (see SETUP_GUIDE.md)")
        print("2. Implement dataset loader in src/data/dataset.py")
        print("3. Start training with: python -m scripts.train --help")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("Common issues:")
        print("- Missing dependencies: Run 'uv pip install -e .'")
        print("- Import errors: Check Python version (>=3.10)")
    
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
