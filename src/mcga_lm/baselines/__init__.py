
from __future__ import annotations

from typing import TYPE_CHECKING

from .keystroke import (
    AdaptiveFrequencyRanker,
    BigramRanker,
    CompletionScanningSimulator,
    GridScanningSimulator,
    KLMParams,
    KeystrokeResult,
)
from .variants import (
    ABLATIONS,
    GENERATIVE_SYSTEMS,
    KEYSTROKE_SYSTEMS,
    VARIANTS,
    Variant,
    get_variant,
)

if TYPE_CHECKING:  # pragma: no cover
    from .intent_classifier import IntentClassifierBaseline, PhraseStore, StoredPhrase

_LAZY = {
    "IntentClassifierBaseline": "intent_classifier",
    "PhraseStore": "intent_classifier",
    "StoredPhrase": "intent_classifier",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "KLMParams",
    "KeystrokeResult",
    "GridScanningSimulator",
    "CompletionScanningSimulator",
    "BigramRanker",
    "AdaptiveFrequencyRanker",
    "IntentClassifierBaseline",
    "PhraseStore",
    "StoredPhrase",
    "VARIANTS",
    "Variant",
    "get_variant",
    "GENERATIVE_SYSTEMS",
    "KEYSTROKE_SYSTEMS",
    "ABLATIONS",
]
