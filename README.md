# Emotion Classification for Social Media Images

A deep learning system that classifies emotions in social media images and suggests appropriate emoji reactions. Built with PyTorch, FastAPI, and modern ML practices.

## 🎯 Overview

This system addresses the problem of inappropriate emoji reactions on social media by:

1. **Analyzing** images using deep learning to detect emotions
2. **Classifying** into 8 discrete emotions (happy, sad, angry, fear, surprise, disgust, neutral, other)
3. **Filtering** emoji reactions - only showing emotionally appropriate ones

## 🏗️ Architecture

### Step A: Image → Emotion Prediction

- **Backbone**: Vision Transformer (ViT) / ResNet50 / EfficientNet
- **Head**: Classification layer with dropout
- **Output**: Multi-label emotion probabilities (BCEWithLogitsLoss)

### Step B: Emotion → Emoji Mapping

- Business rules mapping emotions to allowed/blocked emojis
- Configurable threshold-based filtering
- Union of allowed emojis for multiple detected emotions

## 📊 Supported Emotions

| Emotion  | Allowed Emojis       | Blocked Emojis      |
| -------- | -------------------- | ------------------- |
| Happy    | 😀 😄 😎 😂 ❤️ 👍 🎉 | 😢 😭 😡 😤         |
| Sad      | 😢 😭 💔 🤍 🙏       | 😂 😆 😎 🎉         |
| Angry    | 😡 😤 💢 👎          | 😂 😆 😍            |
| Fear     | 😨 😰 😱 🙏          | 😆 😎 🎉            |
| Disgust  | 🤢 😒 🙄             | 😍 😘 🥰            |
| Surprise | 😮 🤯 😲 😳          | (context-dependent) |
| Neutral  | 👍 🙂 🤝 👌          | 😭 🤯 😡            |
| Other    | 🤔 😕 😬 🤷          | -                   |

## 📁 Project Structure

```
emotion-classification/
├── src/
│   ├── config/           # Configuration files
│   │   ├── emotion_config.py    # Emotion-emoji mappings
│   │   ├── model_config.py      # Model architecture settings
│   │   └── training_config.py   # Training hyperparameters
│   ├── data/             # Data handling
│   │   ├── dataset.py           # PyTorch datasets
│   │   ├── transforms.py        # Image transformations
│   │   └── dataloader.py        # DataLoader utilities
│   ├── models/           # Model architecture
│   │   ├── backbone.py          # Backbone creation (timm)
│   │   └── emotion_classifier.py # Main model class
│   ├── training/         # Training pipeline
│   │   ├── trainer.py           # Training loop
│   │   ├── metrics.py           # Evaluation metrics
│   │   └── loss.py              # Loss functions
│   ├── inference/        # Inference pipeline
│   │   └── predictor.py         # Prediction interface
│   └── api/              # REST API
│       └── app.py               # FastAPI application
├── scripts/              # Executable scripts
│   ├── train.py                 # Training script
│   ├── inference.py             # Inference script
│   └── run_api.py               # API server
├── data/                 # Dataset storage
├── models/               # Saved checkpoints
├── notebooks/            # Jupyter notebooks
├── tests/                # Unit tests
├── main.py               # CLI entry point
└── pyproject.toml        # Dependencies (uv)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- uv package manager

### Installation

```powershell
# Clone or navigate to the project
cd "c:\Users\DananjayaAbey\Desktop\emotion classification"

# Install dependencies using uv
uv pip install -e .

# For development dependencies
uv pip install -e ".[dev]"
```

### Quick Start

```powershell
# Show system info
python main.py info

# Train a model (after preparing your dataset)
python -m scripts.train --data-dir ./data/train --output-dir ./models --epochs 50

# Run inference on an image
python -m scripts.inference --checkpoint ./models/best_model.pth --image test.jpg --show-scores

# Start API server
python -m scripts.run_api --checkpoint ./models/best_model.pth --port 8000
```

## 📚 Recommended Datasets

### 1. **AffectNet** (Faces)

- 1M+ in-the-wild face images
- 440K with emotion labels (7 discrete + valence/arousal)
- Best for: Portrait/selfie heavy content

### 2. **FI (Flickr & Instagram)**

- ~23K social media images
- 8 emotion categories (Mikels' model)
- Best for: General social media content

### 3. **EmoSet** (ICCV 2023)

- Large-scale visual emotion dataset
- Multiple affective labels
- Best for: Comprehensive emotion coverage

## 🎓 Training

### Basic Training

```python
from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.models.emotion_classifier import EmotionClassifier
from src.training.trainer import Trainer

# Configure
model_config = ModelConfig(backbone="vit_base_patch16_224")
training_config = TrainingConfig(
    batch_size=32,
    num_epochs=50,
    learning_rate=1e-4,
)

# Create model
model = EmotionClassifier(model_config)

# Train (after loading data)
# trainer = Trainer(model, train_loader, val_loader, training_config, model_config)
# trainer.train()
```

### Hyperparameters

```python
# Model
- backbone: "vit_base_patch16_224", "resnet50", "efficientnet_b0"
- image_size: 224
- dropout: 0.1

# Training
- batch_size: 32
- learning_rate: 1e-4
- optimizer: AdamW
- scheduler: cosine
- mixed_precision: True
- early_stopping_patience: 10

# Data Augmentation
- horizontal_flip: True
- rotation: 15°
- color_jitter: True
```

## 🔮 Inference

### Python API

```python
from src.inference.predictor import EmotionPredictor
from pathlib import Path

# Load model
predictor = EmotionPredictor.from_checkpoint(
    Path("models/best_model.pth"),
    device="cuda",
    threshold=0.35
)

# Predict
result = predictor.predict("image.jpg")
print(f"Emotions: {result['predicted_emotions']}")
print(f"Allowed emojis: {result['allowed_emojis']}")
```

### REST API

```powershell
# Start server
python -m scripts.run_api --checkpoint ./models/best_model.pth

# Use API (in another terminal or browser)
# Docs: http://localhost:8000/docs
```

**API Endpoints:**

- `POST /predict` - Full emotion prediction with emojis
- `POST /predict_emojis` - Get only allowed emojis
- `GET /emotions` - List supported emotions
- `GET /emotion_mappings` - Get emotion-emoji mappings
- `GET /health` - Health check

**Example cURL:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg" \
  -F "include_scores=true"
```

## 🧪 Model Performance

Track these metrics during training:

- **F1 Score** (macro): Primary metric
- **F1 Score** (micro/weighted): For imbalanced data
- **Per-class F1**: Individual emotion performance
- **Precision/Recall**: Fine-grained analysis

## 🛠️ Development

### Code Style

```powershell
# Format code
uv run black src/ scripts/ tests/

# Sort imports
uv run isort src/ scripts/ tests/

# Lint
uv run flake8 src/ scripts/ tests/

# Type check
uv run mypy src/
```

### Testing

```powershell
# Run tests
uv run pytest tests/

# With coverage
uv run pytest --cov=src tests/
```

## 🎯 Roadmap

- [ ] Implement dataset loaders for AffectNet, FI, EmoSet
- [ ] Add data preprocessing utilities
- [ ] Create training visualization with TensorBoard/Weights & Biases
- [ ] Implement model export (ONNX/TorchScript)
- [ ] Add batch inference script
- [ ] Create web demo with Gradio/Streamlit
- [ ] Add multi-GPU training support
- [ ] Implement test time augmentation (TTA)
- [ ] Add confidence calibration
- [ ] Create Docker container

## 📖 References

- **AffectNet**: [Mollahosseini et al., 2017](http://mohammadmahoor.com/affectnet/)
- **FI Dataset**: [You et al., 2016](https://github.com/dchen236/FairFace)
- **Vision Transformers**: [Dosovitskiy et al., 2021](https://arxiv.org/abs/2010.11929)
- **timm library**: [Ross Wightman](https://github.com/rwightman/pytorch-image-models)

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## 📧 Contact

For questions or suggestions, please open an issue.

---

**Built with ❤️ using PyTorch, FastAPI, and modern ML practices**
