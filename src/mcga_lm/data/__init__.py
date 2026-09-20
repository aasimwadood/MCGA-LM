

from __future__ import annotations

from typing import TYPE_CHECKING

from .corpus import AACIntentCorpus, CorpusExample, build_corpus, realise_reference
from .personas import (
    GroundTruthIntent,
    Persona,
    PersonaSpec,
    PersonaTurn,
    TurnContext,
    build_persona,
    build_persona_suite,
)
from .physiology import FatigueModel, PhysiologySynthesiser, environment_vector
from .taxonomy import PRAGMATIC_FUNCTIONS, SLOT_TYPE

if TYPE_CHECKING:  # pragma: no cover
    from .dataset import SessionTensors, TurnEncoder, sliding_windows

_LAZY = {"SessionTensors": "dataset", "TurnEncoder": "dataset", "sliding_windows": "dataset"}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Persona",
    "PersonaSpec",
    "PersonaTurn",
    "TurnContext",
    "GroundTruthIntent",
    "build_persona",
    "build_persona_suite",
    "FatigueModel",
    "PhysiologySynthesiser",
    "environment_vector",
    "TurnEncoder",
    "SessionTensors",
    "sliding_windows",
    "AACIntentCorpus",
    "CorpusExample",
    "build_corpus",
    "realise_reference",
    "PRAGMATIC_FUNCTIONS",
    "SLOT_TYPE",
]
