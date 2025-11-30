# FI (Flickr & Instagram) Dataset Guide

## 📊 Dataset Overview

**FI Dataset** is currently one of the largest well-labeled emotion datasets for images.

### Key Statistics

- **Total Images**: 23,308 (high-quality labeled)
- **Original Collection**: 90,000 images (before filtering)
- **Source**: Flickr and Instagram
- **Labeling**: 225 Amazon Mechanical Turk (AMT) workers
- **Quality**: Images with at least 3 consistent votes retained
- **Paper**: "Building a large scale dataset for image emotion recognition: The fine print and the benchmark" (AAAI 2016)

### Emotion Categories (Mikels' Model)

The FI dataset uses 8 emotion categories based on Mikels' wheel of emotions:

| FI Emotion  | Our Mapping | Description                     |
| ----------- | ----------- | ------------------------------- |
| Amusement   | Happy       | Funny, entertaining content     |
| Anger       | Angry       | Frustrating, enraging scenes    |
| Awe         | Surprise    | Breathtaking, magnificent views |
| Contentment | Happy       | Peaceful, satisfying moments    |
| Disgust     | Disgust     | Repulsive, unpleasant scenes    |
| Excitement  | Happy       | Thrilling, energetic content    |
| Fear        | Fear        | Scary, threatening situations   |
| Sadness     | Sad         | Depressing, sorrowful scenes    |

---

## 🔗 Download Links

### Official Sources

1. **Author's Homepage**: [https://qzyou.github.io/](https://qzyou.github.io/)

   - Contact: Quanzeng You (quanzeng.you@outlook.com)
   - May require email request for access

2. **Nankai CV Lab**: [http://47.105.62.179:8081/sentiment/index.html](http://47.105.62.179:8081/sentiment/index.html)

   - Alternative download source
   - Multiple emotion datasets available

3. **Paper**: Search "Building a large scale dataset for image emotion recognition" on Google Scholar

---

## 📁 Expected Dataset Structure

After downloading, organize your FI dataset as follows:

```
data/
└── fi_dataset/
    ├── images/
    │   ├── flickr/
    │   │   ├── 123456.jpg
    │   │   ├── 234567.jpg
    │   │   └── ...
    │   └── instagram/
    │       ├── 789012.jpg
    │       ├── 890123.jpg
    │       └── ...
    └── annotations/
        ├── train.txt (or annotations.txt)
        ├── val.txt
        └── test.txt
```

### Annotation Format

**Option 1: Space-separated (train.txt)**

```
123456.jpg amusement
234567.jpg fear
345678.jpg contentment
```

**Option 2: Comma-separated (train.csv)**

```
image_name,emotion
123456.jpg,amusement
234567.jpg,fear
345678.jpg,contentment
```

---

## 🚀 Quick Start

### 1. View Dataset Information

```powershell
python -m scripts.prepare_fi_dataset info
```

### 2. Create Train/Val/Test Splits

If you have a single annotation file:

```powershell
python -m scripts.prepare_fi_dataset create-splits `
    --annotation-file data/fi_dataset/annotations.txt `
    --output-dir data/fi_dataset/annotations `
    --train-ratio 0.7 `
    --val-ratio 0.15 `
    --test-ratio 0.15
```

### 3. Verify Dataset

```powershell
python -m scripts.prepare_fi_dataset verify `
    --data-dir data/fi_dataset `
    --split train
```

### 4. Load Dataset in Python

```python
from pathlib import Path
from src.data.fi_dataset import FIDataset
from src.config.model_config import ModelConfig
from src.data.transforms import get_train_transforms

# Create config and transforms
model_config = ModelConfig()
transform = get_train_transforms(model_config)

# Load dataset
dataset = FIDataset.from_directory(
    data_dir=Path("data/fi_dataset"),
    split="train",
    transform=transform,
    multi_label=True,
)

print(f"Loaded {len(dataset)} images")

# Get a sample
image, label = dataset[0]
print(f"Image shape: {image.shape}")
print(f"Label: {label}")
```

---

## 🎓 Training with FI Dataset

### Update Training Script

Edit `scripts/train.py` to use FI dataset:

```python
from src.data.fi_dataset import FIDataset

# In the training script
train_dataset = FIDataset.from_directory(
    data_dir=Path(args.data_dir),
    split="train",
    transform=train_transform,
)

val_dataset = FIDataset.from_directory(
    data_dir=Path(args.data_dir),
    split="val",
    transform=val_transform,
)
```

### Run Training

```powershell
# Activate environment
.\.venv\Scripts\activate

# Train with FI dataset
python -m scripts.train `
    --data-dir data/fi_dataset `
    --output-dir models/fi_experiment `
    --backbone vit_base_patch16_224 `
    --batch-size 32 `
    --epochs 50 `
    --lr 1e-4 `
    --device cuda
```

---

## 💡 Usage Examples

### Example 1: Load from CSV

```python
from pathlib import Path
from src.data.fi_dataset import FIDataset

dataset = FIDataset.from_csv(
    csv_file=Path("data/fi_dataset/annotations/train.csv"),
    image_dir=Path("data/fi_dataset/images"),
    transform=None,
)
```

### Example 2: Load from Text File

```python
dataset = FIDataset.from_annotation_file(
    annotation_file=Path("data/fi_dataset/annotations/train.txt"),
    image_dir=Path("data/fi_dataset/images"),
    transform=None,
)
```

### Example 3: Create Custom Splits

```python
from src.data.fi_dataset import create_fi_splits

create_fi_splits(
    annotation_file=Path("data/fi_dataset/all_annotations.txt"),
    output_dir=Path("data/fi_dataset/annotations"),
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    shuffle=True,
    seed=42,
)
```

### Example 4: Check Emotion Distribution

```python
from collections import Counter
from src.data.fi_dataset import FIDataset
from src.config.emotion_config import EmotionConfig

# Load dataset
dataset = FIDataset.from_directory(
    Path("data/fi_dataset"),
    split="train",
)

# Count emotions
emotion_counts = Counter()
for _, label in dataset:
    active_emotions = [
        EmotionConfig.IDX_TO_EMOTION[i]
        for i, val in enumerate(label)
        if val > 0
    ]
    for emotion in active_emotions:
        emotion_counts[emotion] += 1

print("Emotion Distribution:")
for emotion, count in emotion_counts.most_common():
    print(f"  {emotion}: {count}")
```

---

## 🔍 Dataset Quality

### Advantages

✅ **Large Scale**: 23,308 high-quality images
✅ **Real Social Media**: Authentic Flickr & Instagram content
✅ **Quality Control**: Multiple AMT worker votes
✅ **8 Emotions**: Comprehensive emotion coverage
✅ **Well-Cited**: AAAI 2016 paper with 500+ citations

### Considerations

⚠️ **Mapping Required**: FI emotions need mapping to your categories
⚠️ **Access**: May require contacting authors
⚠️ **Single Label**: Original dataset has single labels (we convert to multi-label)

---

## 📚 Citation

If you use the FI dataset in your work, please cite:

```bibtex
@inproceedings{you2016building,
  title={Building a large scale dataset for image emotion recognition: The fine print and the benchmark},
  author={You, Quanzeng and Luo, Jiebo and Jin, Hailin and Yang, Jianchao},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={30},
  number={1},
  year={2016}
}
```

---

## 🛠️ Troubleshooting

### Issue: Images Not Found

**Solution**: Check your directory structure

```powershell
# Verify structure
tree data/fi_dataset /F
```

### Issue: Unknown Emotion Labels

**Solution**: Check annotation format

```powershell
# View first few lines
Get-Content data/fi_dataset/annotations/train.txt -Head 10
```

### Issue: Import Errors

**Solution**: Make sure you're in the virtual environment

```powershell
.\.venv\Scripts\activate
```

---

## 🔄 Alternative Datasets

If FI is not available, consider:

1. **AffectNet** (440K images, faces)

   - Good for: Portrait/selfie content
   - Download: http://mohammadmahoor.com/affectnet/

2. **EmoSet** (ICCV 2023, large-scale)

   - Good for: Recent benchmark
   - Check: Latest ICCV papers

3. **Emotion6** (1,980 images)
   - Good for: Quick experiments
   - Download: http://chenlab.ece.cornell.edu/downloads.html

---

## 📞 Support

- Check `src/data/fi_dataset.py` for implementation details
- Run `python -m scripts.prepare_fi_dataset info` for quick reference
- Review the original paper for dataset specifics

---

**Ready to train with FI dataset! 🚀**
