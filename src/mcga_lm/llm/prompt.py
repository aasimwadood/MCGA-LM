

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

from ..config import LLMConfig

INSTRUCTION = "Generate a single, appropriate next utterance that the user would say."

# Sec. 3.5 component 2: the discretised state is "mapped to explicit generation
# instructions". The wording is ours (ASSUMPTION A-11); the low/moderate/high
# split and the shrink-with-fatigue behaviour are the paper's.
STATE_INSTRUCTIONS: Dict[str, str] = {
    "low": (
        "State: low_fatigue. The user has cognitive reserve. Offer an expressive, "
        "fully formed sentence in their own voice."
    ),
    "moderate": (
        "State: moderate_fatigue. Shorten the utterance and prefer familiar phrasing; "
        "drop optional detail."
    ),
    "high": (
        "State: high_fatigue. Emit the shortest utterance that conveys the intent, "
        "using the user's most practised wording. No elaboration."
    ),
}


@dataclass
class UserProfile:
    """Component U of Sec. 3.5."""

    persona_id: str
    diagnosis: str
    age: int
    verbosity: str = "medium"  # low | medium | high
    formality: str = "informal"
    pre_disability_corpus: bool = False  # "drawn from a pre-disability writing corpus if available"
    idiolect: Sequence[str] = field(default_factory=tuple)  # habitual phrases

    def render(self) -> str:
        lines = [
            f"Profile: {self.age}-year-old AAC user, diagnosis: {self.diagnosis}.",
            f"Preferences: verbosity={self.verbosity}, formality={self.formality}.",
        ]
        if self.idiolect:
            lines.append("Habitual phrasing: " + "; ".join(self.idiolect[:5]) + ".")
        if self.pre_disability_corpus:
            lines.append("A pre-disability writing sample is available for voice matching.")
        return " ".join(lines)


@dataclass
class DialogueTurn:
    speaker: str
    text: str


@dataclass
class StructuredPrompt:
    """The assembled prompt ``P_t`` plus the fields a backend may use directly."""

    profile: UserProfile
    state_level: str
    graph_text: str
    history: Sequence[DialogueTurn]
    active_nodes: Sequence[Tuple[str, str, float]] = ()  # (name, node_type, score)
    max_words: int = 14
    use_graph: bool = True

    def render(self) -> str:
        parts = [
            self.profile.render(),
            STATE_INSTRUCTIONS[self.state_level],
        ]
        if self.use_graph:
            parts.append(f"Personal memory: {self.graph_text}")
        parts.append(render_history(self.history))
        parts.append(f"Keep it to at most {self.max_words} words.")
        parts.append(INSTRUCTION)
        return "[INST] " + " ".join(p for p in parts if p) + " [/INST]"


def render_history(history: Sequence[DialogueTurn], max_turns: int = 10) -> str:
    """Component H: the last 10 dialogue turns, with speaker labels (Sec. 3.5)."""
    recent = list(history)[-max_turns:]
    if not recent:
        return "Conversation so far: (none)."
    lines = "; ".join(f"{t.speaker}: {t.text}" for t in recent)
    return f"Conversation so far: {lines}."


def temperature_for_fatigue(fatigue: float, cfg: LLMConfig) -> float:
    """Inverse fatigue modulation of decoding temperature (Sec. 3.5).

    "T = 1.2 at low fatigue for expressive phrasing, decreasing to T = 0.5 at
    high fatigue". ASSUMPTION A-12: linear interpolation in the fatigue index.
    """
    f = min(max(float(fatigue), 0.0), 1.0)
    return cfg.temp_low_fatigue + f * (cfg.temp_high_fatigue - cfg.temp_low_fatigue)


def candidates_for_fatigue(level: str, cfg: LLMConfig) -> int:
    """Option-set size shrinks as reserves deplete (Sec. 1, Sec. 6.1)."""
    return cfg.candidates_by_fatigue[level]


def max_words_for_fatigue(level: str, cfg: LLMConfig) -> int:
    """Verbosity shrinks as reserves deplete (Sec. 1, Sec. 3.5)."""
    return cfg.max_words_by_fatigue[level]


def build_prompt(
    profile: UserProfile,
    state_level: str,
    graph_text: str,
    history: Sequence[DialogueTurn],
    cfg: LLMConfig,
    active_nodes: Optional[Sequence[Tuple[str, str, float]]] = None,
) -> StructuredPrompt:
    return StructuredPrompt(
        profile=profile,
        state_level=state_level,
        graph_text=graph_text if cfg.use_graph_prompt else "",
        history=list(history)[-cfg.history_turns :],
        active_nodes=tuple(active_nodes or ()),
        max_words=max_words_for_fatigue(state_level, cfg),
        use_graph=cfg.use_graph_prompt,
    )
