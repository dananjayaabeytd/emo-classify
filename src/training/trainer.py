"""Training loop implementation."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Optional, Dict

from ..models.emotion_classifier import EmotionClassifier
from ..config.training_config import TrainingConfig
from ..config.model_config import ModelConfig
from .loss import get_loss_function
from .metrics import EmotionMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for emotion classification model."""

    def __init__(
        self,
        model: EmotionClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        training_config: TrainingConfig,
        model_config: ModelConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize trainer.

        Args:
            model: Emotion classification model
            train_loader: Training data loader
            val_loader: Validation data loader
            training_config: Training configuration
            model_config: Model configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.training_config = training_config
        self.model_config = model_config
        self.device = device

        # Loss function
        self.criterion = get_loss_function(model_config.multi_label)

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision training
        self.scaler = GradScaler() if training_config.mixed_precision else None

        # Metrics
        self.train_metrics = EmotionMetrics(
            num_classes=model_config.num_classes,
            emotion_labels=training_config.data_dir.name.split("/")[-1] if hasattr(training_config.data_dir, 'name') else None
        )
        self.val_metrics = EmotionMetrics(
            num_classes=model_config.num_classes,
            emotion_labels=training_config.data_dir.name.split("/")[-1] if hasattr(training_config.data_dir, 'name') else None
        )

        # Tracking
        self.current_epoch = 0
        self.best_val_f1 = 0.0
        self.patience_counter = 0

        logger.info(f"Trainer initialized on device: {device}")
        logger.info(f"Model info: {model.get_model_info()}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        if self.training_config.optimizer.lower() == "adamw":
            return AdamW(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay,
                betas=self.training_config.betas,
            )
        elif self.training_config.optimizer.lower() == "sgd":
            return SGD(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.training_config.optimizer}")

    def _create_scheduler(self):
        """Create learning rate scheduler based on config."""
        if self.training_config.scheduler.lower() == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.training_config.num_epochs,
            )
        elif self.training_config.scheduler.lower() == "step":
            return StepLR(
                self.optimizer,
                step_size=self.training_config.lr_decay_epochs,
                gamma=self.training_config.lr_decay_factor,
            )
        elif self.training_config.scheduler.lower() == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=self.training_config.lr_decay_factor,
                patience=5,
            )
        else:
            return None

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()

        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Forward pass with mixed precision
            if self.scaler:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)

                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Update metrics
            predictions, _ = self.model.predict_emotions(images)
            self.train_metrics.update(predictions, targets)

            total_loss += loss.item()

            # Update progress bar
            if batch_idx % self.training_config.log_every == 0:
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Compute metrics
        avg_loss = total_loss / len(self.train_loader)
        metrics = self.train_metrics.compute(self.model_config.multi_label)
        metrics["loss"] = avg_loss

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        self.val_metrics.reset()

        total_loss = 0.0

        for images, targets in tqdm(self.val_loader, desc="Validation"):
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            logits = self.model(images)
            loss = self.criterion(logits, targets)

            # Update metrics
            predictions, _ = self.model.predict_emotions(images)
            self.val_metrics.update(predictions, targets)

            total_loss += loss.item()

        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        metrics = self.val_metrics.compute(self.model_config.multi_label)
        metrics["loss"] = avg_loss

        return metrics

    def train(self):
        """Full training loop."""
        logger.info("Starting training...")

        for epoch in range(self.training_config.num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            logger.info(
                f"Epoch {epoch + 1}/{self.training_config.num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train F1: {train_metrics.get('f1_macro', 0):.4f}"
            )

            # Validate
            if (epoch + 1) % self.training_config.eval_every == 0:
                val_metrics = self.validate()
                logger.info(
                    f"Validation - Loss: {val_metrics['loss']:.4f}, "
                    f"F1: {val_metrics.get('f1_macro', 0):.4f}"
                )

                # Update scheduler
                if self.scheduler:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics.get("f1_macro", 0))
                    else:
                        self.scheduler.step()

                # Save checkpoint
                current_f1 = val_metrics.get("f1_macro", 0)
                if current_f1 > self.best_val_f1:
                    self.best_val_f1 = current_f1
                    self.save_checkpoint("best_model.pth")
                    logger.info(f"New best model saved! F1: {current_f1:.4f}")
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                # Early stopping
                if self.patience_counter >= self.training_config.early_stopping_patience:
                    logger.info("Early stopping triggered!")
                    break

            # Save periodic checkpoint
            if (epoch + 1) % self.training_config.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")

        logger.info("Training completed!")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = self.training_config.output_dir / filename
        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_f1": self.best_val_f1,
                "model_config": self.model_config,
            },
            checkpoint_path,
        )
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_f1 = checkpoint["best_val_f1"]
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
