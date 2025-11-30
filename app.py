"""
Simple Streamlit UI for Emotion Classification
Run with: streamlit run app.py
"""

import streamlit as st
import torch
from PIL import Image
import requests
from io import BytesIO
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.inference.predictor import EmotionPredictor
from src.config.emotion_config import EmotionConfig

# Page config
st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="😊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emotion-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
    .emoji-large {
        font-size: 3rem;
        text-align: center;
    }
    .score-bar {
        height: 25px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(checkpoint_path):
    """Load the trained model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = EmotionPredictor.from_checkpoint(
        checkpoint_path=Path(checkpoint_path),
        device=device,
        threshold=0.35
    )
    return predictor

def main():
    # Header
    st.markdown('<h1 class="main-header">😊 Emotion Classification & Emoji Filter 🎭</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        checkpoint_path = st.text_input(
            "Model Checkpoint Path",
            value="models/best_model.pth"
        )
        
        threshold = st.slider(
            "Emotion Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.05
        )
        
        st.markdown("---")
        st.markdown("### 📊 About")
        st.markdown("""
        This app classifies emotions from images and recommends appropriate emojis for social media reactions.
        
        **Emotions:**
        - 😊 Happy
        - 😢 Sad
        - 😡 Angry
        - 😨 Fear
        - 😮 Surprise
        - 🤢 Disgust
        - 😐 Neutral
        - 🤔 Other
        """)
        
        device = "🖥️ CUDA" if torch.cuda.is_available() else "💻 CPU"
        st.info(f"Running on: {device}")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # Upload options
        upload_option = st.radio(
            "Choose input method:",
            ["Upload Image", "Use Webcam", "Image URL", "Sample Images"]
        )
        
        image = None
        
        if upload_option == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=["jpg", "jpeg", "png"]
            )
            if uploaded_file:
                image = Image.open(uploaded_file)
        
        elif upload_option == "Use Webcam":
            camera_image = st.camera_input("Take a picture")
            if camera_image:
                image = Image.open(camera_image)
        
        elif upload_option == "Image URL":
            url = st.text_input("Enter image URL:")
            if url:
                try:
                    response = requests.get(url)
                    image = Image.open(BytesIO(response.content))
                except Exception as e:
                    st.error(f"Failed to load image: {e}")
        
        elif upload_option == "Sample Images":
            # Get sample images from test dataset
            test_dir = Path("data/fer2013/test")
            if test_dir.exists():
                emotions = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]
                samples = {}
                for emotion in emotions:
                    emotion_dir = test_dir / emotion
                    if emotion_dir.exists():
                        images = list(emotion_dir.glob("*.jpg"))
                        if images:
                            samples[emotion] = images[0]
                
                if samples:
                    selected_emotion = st.selectbox("Select sample emotion:", list(samples.keys()))
                    if st.button("Load Sample"):
                        image = Image.open(samples[selected_emotion])
            else:
                st.warning("No sample images found. Upload your own image!")
        
        if image:
            st.image(image, caption="Input Image", use_container_width=True)
    
    with col2:
        st.header("🎯 Results")
        
        if image:
            try:
                # Ensure image is RGB (convert grayscale to RGB)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                # Load model
                with st.spinner("Loading model..."):
                    predictor = load_model(checkpoint_path)
                    predictor.threshold = threshold
                
                # Predict
                with st.spinner("Analyzing emotion..."):
                    result = predictor.predict(image, return_probabilities=True)
                
                # Display predicted emotions
                st.subheader("🎭 Detected Emotions")
                predicted_emotions = result["predicted_emotions"]
                
                if predicted_emotions:
                    emotion_cols = st.columns(len(predicted_emotions))
                    for idx, emotion in enumerate(predicted_emotions):
                        with emotion_cols[idx]:
                            # Get emoji for emotion
                            emoji_config = EmotionConfig()
                            emoji = emoji_config.EMOTION_TO_EMOJIS.get(emotion, ["🤔"])[0]
                            st.markdown(f'<div class="emoji-large">{emoji}</div>', unsafe_allow_html=True)
                            st.markdown(f'<p style="text-align: center; font-weight: bold;">{emotion.upper()}</p>', unsafe_allow_html=True)
                else:
                    st.info("No strong emotions detected")
                
                # Display allowed emojis
                st.subheader("✅ Allowed Emojis")
                allowed_emojis = result["allowed_emojis"]
                if allowed_emojis:
                    emoji_html = " ".join([f'<span style="font-size: 2rem; margin: 0.2rem;">{emoji}</span>' for emoji in allowed_emojis])
                    st.markdown(f'<div style="text-align: center; padding: 1rem;">{emoji_html}</div>', unsafe_allow_html=True)
                else:
                    st.info("No emoji restrictions")
                
                # Display emotion scores
                st.subheader("📊 Emotion Scores")
                scores = result["emotion_scores"]
                
                # Sort by score
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                
                for emotion, score in sorted_scores:
                    col_name, col_bar = st.columns([1, 3])
                    with col_name:
                        st.write(f"**{emotion.capitalize()}**")
                    with col_bar:
                        st.progress(float(score))
                        st.caption(f"{score:.2%}")
                
                # Additional info
                st.markdown("---")
                st.info(f"**Confidence Threshold:** {threshold}")
                
            except Exception as e:
                st.error(f"Error processing image: {e}")
                st.exception(e)
        else:
            st.info("👈 Upload or select an image to start")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; padding: 1rem;">
        <p>Built with PyTorch, ResNet50, and FER2013 Dataset</p>
        <p>Emotion → Emoji mapping for social media content filtering</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
