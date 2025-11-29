"""Visualization utilities for emotion classification."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
from pathlib import Path


def plot_emotion_distribution(
    emotion_counts: Dict[str, int],
    save_path: Path = None,
    title: str = "Emotion Distribution in Dataset",
):
    """
    Plot the distribution of emotions in the dataset.

    Args:
        emotion_counts: Dictionary mapping emotion names to counts
        save_path: Optional path to save the plot
        title: Plot title
    """
    emotions = list(emotion_counts.keys())
    counts = list(emotion_counts.values())

    plt.figure(figsize=(12, 6))
    bars = plt.bar(emotions, counts, color="skyblue", edgecolor="navy", alpha=0.7)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    plt.xlabel("Emotion", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Path = None,
    metrics: List[str] = None,
):
    """
    Plot training history (loss, metrics over epochs).

    Args:
        history: Dictionary with keys like 'train_loss', 'val_loss', 'train_f1', 'val_f1'
        save_path: Optional path to save the plot
        metrics: List of metrics to plot (default: all)
    """
    if metrics is None:
        metrics = list(history.keys())

    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))

    if num_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        if metric in history:
            axes[idx].plot(history[metric], label=metric, linewidth=2)
            axes[idx].set_xlabel("Epoch")
            axes[idx].set_ylabel(metric.replace("_", " ").title())
            axes[idx].set_title(f"{metric.replace('_', ' ').title()} Over Epochs")
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Path = None,
    normalize: bool = False,
):
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_path: Optional path to save the plot
        normalize: Whether to normalize the matrix
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add text annotations
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_emotion_scores(
    emotion_scores: Dict[str, float],
    threshold: float = 0.35,
    save_path: Path = None,
):
    """
    Plot emotion prediction scores with threshold line.

    Args:
        emotion_scores: Dictionary mapping emotions to scores
        threshold: Classification threshold
        save_path: Optional path to save the plot
    """
    emotions = list(emotion_scores.keys())
    scores = list(emotion_scores.values())

    plt.figure(figsize=(12, 6))
    colors = ["green" if s >= threshold else "lightcoral" for s in scores]
    bars = plt.bar(emotions, scores, color=colors, edgecolor="navy", alpha=0.7)

    # Add threshold line
    plt.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold ({threshold})")

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
        )

    plt.xlabel("Emotion", fontsize=12)
    plt.ylabel("Confidence Score", fontsize=12)
    plt.title("Emotion Prediction Scores", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()
