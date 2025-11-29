"""Emotion labels and emoji mappings configuration."""

from typing import Dict, List


class EmotionConfig:
    """Configuration for emotion labels and their corresponding emoji mappings."""

    # 8 discrete emotions based on AffectNet / FI datasets
    EMOTIONS: List[str] = [
        "happy",
        "sad",
        "angry",
        "fear",
        "surprise",
        "disgust",
        "neutral",
        "other",
    ]

    # Map each emotion to allowed emojis
    EMOTION_TO_EMOJIS: Dict[str, List[str]] = {
        "happy": ["😀", "😄", "😎", "😂", "❤️", "👍", "🎉", "😊", "🥰", "😍", "🤗"],
        "sad": ["😢", "😭", "💔", "🤍", "🤗", "🙏", "😔", "😞", "☹️"],
        "angry": ["😡", "😤", "💢", "👎", "😠", "🤬", "😾"],
        "fear": ["😨", "😰", "😱", "🙏", "😧", "😦", "😟"],
        "surprise": ["😮", "🤯", "😲", "😳", "🙀", "😯"],
        "disgust": ["🤢", "😒", "🙄", "🤮", "😖", "🤭"],
        "neutral": ["👍", "🙂", "🤝", "👌", "😐", "😶"],
        "other": ["🤔", "😕", "😬", "🤷"],
    }

    # Typically blocked emojis for each emotion (inverse mapping)
    EMOTION_BLOCKED_EMOJIS: Dict[str, List[str]] = {
        "happy": ["😢", "😭", "😡", "😤", "😰", "😱", "🤢"],
        "sad": ["😂", "😆", "😎", "🎉", "🤣", "😁"],
        "angry": ["😂", "😆", "😍", "🥰", "😘", "🎉"],
        "fear": ["😆", "😎", "🎉", "😂", "🤣"],
        "disgust": ["😍", "😘", "🥰", "❤️", "😂"],
        "surprise": [],  # Mostly neutral, context-dependent
        "neutral": ["😭", "🤯", "😡", "😱"],
        "other": [],
    }

    # Emotion index mapping
    EMOTION_TO_IDX: Dict[str, int] = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}
    IDX_TO_EMOTION: Dict[int, str] = {idx: emotion for idx, emotion in enumerate(EMOTIONS)}

    # Number of emotion classes
    NUM_EMOTIONS: int = len(EMOTIONS)

    # Threshold for multi-label classification
    EMOTION_THRESHOLD: float = 0.35

    @classmethod
    def get_allowed_emojis(cls, emotions: List[str]) -> List[str]:
        """
        Get the union of allowed emojis for given emotions.

        Args:
            emotions: List of predicted emotion labels

        Returns:
            List of allowed emoji strings
        """
        allowed = set()
        for emotion in emotions:
            if emotion in cls.EMOTION_TO_EMOJIS:
                allowed.update(cls.EMOTION_TO_EMOJIS[emotion])
        return sorted(list(allowed))

    @classmethod
    def get_blocked_emojis(cls, emotions: List[str]) -> List[str]:
        """
        Get the union of blocked emojis for given emotions.

        Args:
            emotions: List of predicted emotion labels

        Returns:
            List of blocked emoji strings
        """
        blocked = set()
        for emotion in emotions:
            if emotion in cls.EMOTION_BLOCKED_EMOJIS:
                blocked.update(cls.EMOTION_BLOCKED_EMOJIS[emotion])
        return sorted(list(blocked))
