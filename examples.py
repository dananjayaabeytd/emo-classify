"""Example usage of the emotion classification system."""

# Example 1: Basic emotion prediction
def example_basic_prediction():
    """Basic emotion prediction from an image."""
    from src.inference.predictor import EmotionPredictor
    from pathlib import Path
    
    # Load predictor from checkpoint
    predictor = EmotionPredictor.from_checkpoint(
        Path("models/best_model.pth"),
        device="cuda",  # or "cpu"
        threshold=0.35
    )
    
    # Predict emotions
    result = predictor.predict("path/to/image.jpg", return_probabilities=True)
    
    print("Predicted Emotions:", result["predicted_emotions"])
    print("Allowed Emojis:", result["allowed_emojis"])
    print("\nEmotion Scores:")
    for emotion, score in result["emotion_scores"].items():
        print(f"  {emotion}: {score:.3f}")


# Example 2: Batch prediction
def example_batch_prediction():
    """Process multiple images at once."""
    from src.inference.predictor import EmotionPredictor
    from pathlib import Path
    
    predictor = EmotionPredictor.from_checkpoint(Path("models/best_model.pth"))
    
    images = ["image1.jpg", "image2.jpg", "image3.jpg"]
    results = predictor.predict_batch(images)
    
    for i, result in enumerate(results):
        print(f"\nImage {i+1}:")
        print(f"  Emotions: {result['predicted_emotions']}")
        print(f"  Emojis: {' '.join(result['allowed_emojis'])}")


# Example 3: Training a model
def example_training():
    """Train an emotion classification model."""
    from src.config.model_config import ModelConfig
    from src.config.training_config import TrainingConfig
    from src.models.emotion_classifier import EmotionClassifier
    from src.training.trainer import Trainer
    from pathlib import Path
    
    # Configure
    model_config = ModelConfig(
        backbone="vit_base_patch16_224",
        num_classes=8,
        multi_label=True,
    )
    
    training_config = TrainingConfig(
        data_dir=Path("data"),
        output_dir=Path("models"),
        batch_size=32,
        num_epochs=50,
        learning_rate=1e-4,
    )
    
    # Create model
    model = EmotionClassifier(model_config)
    print(f"Model created: {model.get_model_info()}")
    
    # Load your data and create dataloaders
    # train_loader, val_loader = ...
    
    # Create trainer and train
    # trainer = Trainer(model, train_loader, val_loader, training_config, model_config)
    # trainer.train()


# Example 4: Using the API
def example_api_usage():
    """Example API requests."""
    import requests
    
    # Start the API server first:
    # python -m scripts.run_api --checkpoint models/best_model.pth
    
    url = "http://localhost:8000/predict"
    
    # Upload image
    with open("test_image.jpg", "rb") as f:
        files = {"file": f}
        params = {"include_scores": True}
        response = requests.post(url, files=files, params=params)
    
    result = response.json()
    print("API Response:")
    print(f"  Emotions: {result['predicted_emotions']}")
    print(f"  Emojis: {result['allowed_emojis']}")


# Example 5: Custom emotion-emoji mapping
def example_custom_mapping():
    """Use custom emotion-emoji mappings."""
    from src.config.emotion_config import EmotionConfig
    
    # View default mappings
    print("Happy emojis:", EmotionConfig.EMOTION_TO_EMOJIS["happy"])
    
    # Get allowed emojis for multiple emotions
    emotions = ["happy", "surprise"]
    allowed = EmotionConfig.get_allowed_emojis(emotions)
    print(f"\nAllowed emojis for {emotions}:")
    print(" ".join(allowed))
    
    # Get blocked emojis
    blocked = EmotionConfig.get_blocked_emojis(emotions)
    print(f"\nBlocked emojis for {emotions}:")
    print(" ".join(blocked))


# Example 6: Model inference with custom threshold
def example_custom_threshold():
    """Use different thresholds for emotion detection."""
    from src.inference.predictor import EmotionPredictor
    from pathlib import Path
    
    # More permissive (detects more emotions)
    predictor_low = EmotionPredictor.from_checkpoint(
        Path("models/best_model.pth"),
        threshold=0.25
    )
    
    # More strict (detects fewer emotions)
    predictor_high = EmotionPredictor.from_checkpoint(
        Path("models/best_model.pth"),
        threshold=0.50
    )
    
    image = "test_image.jpg"
    
    result_low = predictor_low.predict(image)
    result_high = predictor_high.predict(image)
    
    print("Low threshold (0.25):", result_low["predicted_emotions"])
    print("High threshold (0.50):", result_high["predicted_emotions"])


if __name__ == "__main__":
    print("Emotion Classification System - Examples")
    print("=" * 50)
    print("\nNote: Make sure you have a trained model checkpoint before running these examples.")
    print("\nAvailable examples:")
    print("  1. Basic prediction")
    print("  2. Batch prediction")
    print("  3. Training a model")
    print("  4. Using the API")
    print("  5. Custom emotion-emoji mapping")
    print("  6. Custom threshold")
    print("\nUncomment the function calls below to run specific examples.")
    
    # Uncomment to run examples:
    # example_basic_prediction()
    # example_batch_prediction()
    # example_training()
    # example_api_usage()
    # example_custom_mapping()
    # example_custom_threshold()
