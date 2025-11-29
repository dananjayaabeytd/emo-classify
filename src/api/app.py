"""FastAPI application for emotion classification service."""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import io
from PIL import Image

from ..inference.predictor import EmotionPredictor
from ..config.model_config import ModelConfig


class PredictionResponse(BaseModel):
    """Response model for emotion predictions."""

    predicted_emotions: List[str]
    allowed_emojis: List[str]
    emotion_scores: Optional[dict] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool


def create_app(
    model_checkpoint_path: Path,
    device: str = "cuda",
    threshold: float = 0.35,
) -> FastAPI:
    """
    Create FastAPI application.

    Args:
        model_checkpoint_path: Path to trained model checkpoint
        device: Device to run inference on
        threshold: Threshold for multi-label classification

    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="Emotion Classification API",
        description="API for classifying emotions in social media images and suggesting allowed emojis",
        version="0.1.0",
    )

    # Load model
    try:
        predictor = EmotionPredictor.from_checkpoint(
            model_checkpoint_path, device=device, threshold=threshold
        )
        model_loaded = True
    except Exception as e:
        print(f"Error loading model: {e}")
        predictor = None
        model_loaded = False

    @app.get("/", response_model=HealthResponse)
    async def root():
        """Root endpoint - health check."""
        return HealthResponse(status="ok", model_loaded=model_loaded)

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return HealthResponse(status="ok", model_loaded=model_loaded)

    @app.post("/predict", response_model=PredictionResponse)
    async def predict_emotion(
        file: UploadFile = File(...),
        include_scores: bool = False,
    ):
        """
        Predict emotions from an uploaded image.

        Args:
            file: Image file upload
            include_scores: Whether to include emotion probability scores

        Returns:
            Prediction response with emotions and allowed emojis
        """
        if not model_loaded or predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        try:
            # Read image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")

            # Predict
            result = predictor.predict(image, return_probabilities=include_scores)

            return PredictionResponse(**result)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    @app.post("/predict_emojis")
    async def predict_emojis_only(file: UploadFile = File(...)):
        """
        Get only the allowed emojis for an image (simplified endpoint).

        Args:
            file: Image file upload

        Returns:
            JSON response with allowed emojis
        """
        if not model_loaded or predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        try:
            # Read image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")

            # Get emojis
            emojis = predictor.get_allowed_emojis_only(image)

            return JSONResponse(content={"emojis": emojis})

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    @app.get("/emotions")
    async def get_emotions():
        """Get list of supported emotions."""
        from ..config.emotion_config import EmotionConfig

        return JSONResponse(
            content={
                "emotions": EmotionConfig.EMOTIONS,
                "num_emotions": EmotionConfig.NUM_EMOTIONS,
            }
        )

    @app.get("/emotion_mappings")
    async def get_emotion_mappings():
        """Get emotion to emoji mappings."""
        from ..config.emotion_config import EmotionConfig

        return JSONResponse(content={"mappings": EmotionConfig.EMOTION_TO_EMOJIS})

    return app
