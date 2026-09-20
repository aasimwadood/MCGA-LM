"""Phase IV -- retrieval-augmented intent synthesis (paper Sec. 3.5)."""

from .base import GeneratedUtterance, LanguageBackend
from .hf_backend import build_backend, check_embedding_dim
from .prompt import (
    STATE_INSTRUCTIONS,
    DialogueTurn,
    StructuredPrompt,
    UserProfile,
    build_prompt,
    candidates_for_fatigue,
    max_words_for_fatigue,
    temperature_for_fatigue,
)
from .template_backend import TemplateLanguageBackend

__all__ = [
    "GeneratedUtterance",
    "LanguageBackend",
    "TemplateLanguageBackend",
    "build_backend",
    "check_embedding_dim",
    "StructuredPrompt",
    "UserProfile",
    "DialogueTurn",
    "build_prompt",
    "temperature_for_fatigue",
    "candidates_for_fatigue",
    "max_words_for_fatigue",
    "STATE_INSTRUCTIONS",
]
