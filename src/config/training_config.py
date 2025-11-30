"""Training configuration."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration for training the emotion classification model."""

    # Paths
    data_dir: Path = Path("data")
    output_dir: Path = Path("models")
    log_dir: Path = Path("logs")

    # Training hyperparameters
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 5

    # Optimizer
    optimizer: str = "adamw"
    betas: tuple = (0.9, 0.999)

    # Learning rate scheduler
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    lr_decay_factor: float = 0.1
    lr_decay_epochs: int = 10

    # Data split
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

    # Data augmentation (training only)
    use_augmentation: bool = True
    horizontal_flip: bool = True
    rotation_degrees: int = 15
    color_jitter: bool = True
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2

    # Training settings
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True  # Use automatic mixed precision (AMP)

    # Checkpointing
    save_every: int = 5  # Save checkpoint every N epochs
    save_best_only: bool = True
    early_stopping_patience: int = 10

    # Logging
    log_every: int = 50  # Log every N batches
    eval_every: int = 1  # Evaluate every N epochs

    # Random seed
    seed: int = 42

    def __post_init__(self):
        """Create directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
