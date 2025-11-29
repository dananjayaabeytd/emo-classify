# Setup and Next Steps Guide

## ✅ Project Setup Complete!

Your emotion classification system has been successfully set up with a clean, modular architecture.

## 📦 What's Been Created

### Core Modules

- **`src/config/`** - All configuration files (emotions, model, training)
- **`src/data/`** - Dataset handling and preprocessing
- **`src/models/`** - Model architecture (ViT, ResNet, EfficientNet support)
- **`src/training/`** - Training loop, metrics, and loss functions
- **`src/inference/`** - Prediction interface
- **`src/api/`** - FastAPI REST API

### Scripts

- **`scripts/train.py`** - Training script
- **`scripts/inference.py`** - Inference script
- **`scripts/run_api.py`** - API server

### Configuration

- **`pyproject.toml`** - All dependencies managed by uv
- **`requirements.txt`** - Quick install reference
- **`.gitignore`** - Proper Python gitignore

## 🚀 Quick Start

### 1. Verify Installation

```powershell
# Activate virtual environment
.\.venv\Scripts\activate

# Test the system
python main.py info
```

### 2. Get Datasets

You need to download one or more of these datasets:

#### **Option A: FI (Flickr & Instagram) - Recommended for beginners**

- URL: Check the paper "Building a Large Scale Dataset for Image Emotion Recognition"
- Size: ~23K images
- Format: Images + CSV with emotion labels
- Best for: Social media style images

#### **Option B: AffectNet**

- URL: http://mohammadmahoor.com/affectnet/
- Size: 440K labeled images
- Format: Images + labels
- Best for: Face-heavy content

#### **Option C: EmoSet**

- URL: Check ICCV 2023 paper
- Size: Large-scale
- Best for: Comprehensive coverage

### 3. Prepare Your Dataset

Create a dataset loader for your chosen dataset. Example structure:

```
data/
├── train/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── val/
│   └── ...
├── test/
│   └── ...
└── annotations.csv  # or annotations.json
```

### 4. Implement Dataset Loader

Edit `src/data/dataset.py` to implement your dataset loading:

```python
# Example for CSV format:
# image_path,happy,sad,angry,fear,surprise,disgust,neutral,other
# train/img_001.jpg,1,0,0,0,0,0,0,0
# train/img_002.jpg,0,1,0,0,0,0,0,0

@classmethod
def from_csv(cls, csv_path, image_dir, transform=None):
    import pandas as pd
    df = pd.read_csv(csv_path)

    image_paths = [Path(image_dir) / p for p in df['image_path']]
    labels = df[EmotionConfig.EMOTIONS].values

    return cls(image_paths, labels, transform)
```

## 🎯 Next Steps

### Phase 1: Data Preparation (Week 1)

1. ✅ Download dataset(s)
2. ✅ Implement dataset loader in `src/data/dataset.py`
3. ✅ Verify data loading works
4. ✅ Check label distribution (class imbalance)

### Phase 2: Training Setup (Week 1-2)

1. ✅ Start with small backbone (ResNet50) for quick experimentation
2. ✅ Train on subset of data to verify pipeline
3. ✅ Monitor metrics (F1 score, per-class performance)
4. ✅ Adjust hyperparameters

### Phase 3: Model Training (Week 2-3)

1. ✅ Train with full dataset
2. ✅ Try different backbones (ViT, EfficientNet)
3. ✅ Implement data augmentation tuning
4. ✅ Handle class imbalance (weighted loss, focal loss)

### Phase 4: Evaluation & Tuning (Week 3-4)

1. ✅ Evaluate on test set
2. ✅ Analyze per-emotion performance
3. ✅ Tune threshold for multi-label classification
4. ✅ Create confusion matrix visualization

### Phase 5: Deployment (Week 4+)

1. ✅ Export best model
2. ✅ Test API endpoints
3. ✅ Create simple web demo
4. ✅ Document usage

## 💻 Development Workflow

### Training a Model

```powershell
# Basic training
python -m scripts.train `
    --data-dir ./data/train `
    --output-dir ./models `
    --backbone vit_base_patch16_224 `
    --batch-size 32 `
    --epochs 50 `
    --lr 1e-4

# With GPU
python -m scripts.train `
    --data-dir ./data/train `
    --output-dir ./models `
    --device cuda

# Resume from checkpoint
python -m scripts.train `
    --checkpoint ./models/checkpoint_epoch_20.pth `
    --resume
```

### Running Inference

```powershell
# Single image
python -m scripts.inference `
    --checkpoint ./models/best_model.pth `
    --image test_image.jpg `
    --show-scores

# Batch inference
python -m scripts.inference `
    --checkpoint ./models/best_model.pth `
    --image-dir ./test_images/ `
    --output results.json
```

### Starting API Server

```powershell
# Development server
python -m scripts.run_api `
    --checkpoint ./models/best_model.pth `
    --port 8000 `
    --reload

# Production server
python -m scripts.run_api `
    --checkpoint ./models/best_model.pth `
    --host 0.0.0.0 `
    --port 8000
```

## 🔧 Configuration Tips

### Model Configuration (`src/config/model_config.py`)

```python
# For quick experiments (fast training)
backbone = "resnet50"
image_size = 224
batch_size = 64

# For best performance (slower training)
backbone = "vit_base_patch16_224"
image_size = 224
batch_size = 32
```

### Training Configuration (`src/config/training_config.py`)

```python
# Quick experimentation
num_epochs = 10
learning_rate = 3e-4
save_every = 2

# Full training
num_epochs = 50
learning_rate = 1e-4
save_every = 5
early_stopping_patience = 10
```

### Emotion-Emoji Mapping (`src/config/emotion_config.py`)

You can customize the emoji mappings based on your platform:

```python
EMOTION_TO_EMOJIS = {
    "happy": ["😀", "😄", "😊", "..."],  # Add your platform's emojis
    # ...
}
```

## 📊 Monitoring Training

### Option 1: Console Output

- Loss and F1 scores printed during training
- Per-epoch validation metrics

### Option 2: TensorBoard (TODO - Add integration)

```powershell
tensorboard --logdir logs/
```

### Option 3: Weights & Biases (TODO - Add integration)

```python
# In trainer.py
import wandb
wandb.init(project="emotion-classification")
```

## 🐛 Troubleshooting

### Out of Memory (OOM)

```python
# Reduce batch size
training_config.batch_size = 16  # or 8

# Use gradient accumulation
# (TODO: Add to trainer)

# Use mixed precision (already enabled)
training_config.mixed_precision = True
```

### Slow Training

```python
# Reduce image size
model_config.image_size = 192  # instead of 224

# Use smaller backbone
model_config.backbone = "resnet50"  # instead of ViT

# Reduce workers
training_config.num_workers = 2
```

### Class Imbalance

```python
# Use weighted loss
from src.training.loss import WeightedBCEWithLogitsLoss

# Or focal loss
from src.training.loss import FocalLoss
```

## 📝 Code Quality

```powershell
# Format code
uv run black src/ scripts/

# Sort imports
uv run isort src/ scripts/

# Lint
uv run flake8 src/ scripts/

# Type check
uv run mypy src/
```

## 🎓 Learning Resources

### Papers

- Vision Transformers: https://arxiv.org/abs/2010.11929
- AffectNet: Check Mohammad Mahoor's website
- Multi-label Classification: https://arxiv.org/abs/1809.02352

### Code References

- timm library: https://github.com/rwightman/pytorch-image-models
- PyTorch examples: https://github.com/pytorch/examples
- FastAPI docs: https://fastapi.tiangolo.com

## 🤝 Contributing

To add new features:

1. Create a new branch
2. Add your feature in the appropriate module
3. Update tests and documentation
4. Run linting and tests
5. Submit PR

## 📧 Getting Help

If you encounter issues:

1. Check this guide
2. Review the code documentation
3. Check GitHub issues (if using version control)
4. Consult the README.md

## ✨ Example Workflow

Here's a complete example workflow:

```powershell
# 1. Activate environment
.\.venv\Scripts\activate

# 2. Check system info
python main.py info

# 3. Prepare your dataset (manually)
# Download and organize in data/ folder

# 4. Implement dataset loader (edit src/data/dataset.py)

# 5. Train model
python -m scripts.train --data-dir ./data --epochs 50

# 6. Test inference
python -m scripts.inference --checkpoint ./models/best_model.pth --image test.jpg

# 7. Start API
python -m scripts.run_api --checkpoint ./models/best_model.pth

# 8. Test API
# Open browser: http://localhost:8000/docs
```

## 🎯 Success Metrics

Your system is working well when:

- ✅ Training loss decreases steadily
- ✅ Validation F1 score > 0.6 (good), > 0.7 (excellent)
- ✅ Per-class F1 scores are balanced
- ✅ API responds < 500ms for inference
- ✅ Predicted emojis match human intuition

---

**You're ready to build! Start with Phase 1 (Data Preparation). Good luck! 🚀**
