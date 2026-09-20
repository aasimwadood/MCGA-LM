

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

from ..config import Config

# Systems that share the MCGA-LM forward pass and differ only by configuration.
GENERATIVE_SYSTEMS = ("MCGA-LM", "LLM-Only", "M-LLM", "RAG-LLM", "Non-LLM-Intent")
KEYSTROKE_SYSTEMS = ("TouchChat", "Static-WP-bigram", "Adaptive-grid")
ABLATIONS = ("\\PerceiverIO", "\\TFT", "\\GAT", "\\BayesianGate", "\\cross-attention")


@dataclass
class Variant:
    """A named configuration transform, plus flags the runner needs."""

    name: str
    apply: Callable[[Config], Config]
    ablate_perceiver: bool = False
    fixed_temperature: bool = False  # RAG-LLM ignores the fatigue index
    note: str = ""


def _clone(cfg: Config) -> Config:
    return copy.deepcopy(cfg)


def full_system(cfg: Config) -> Config:
    return _clone(cfg)


def llm_only(cfg: Config) -> Config:
    """No memory graph, no fatigue adaptation, no gate (Sec. 4.3 baseline 2)."""
    c = _clone(cfg)
    c.graph.enabled = False
    c.llm.use_graph_prompt = False
    c.tft.enabled = False
    c.safety.enabled = False
    return c


def m_llm(cfg: Config) -> Config:
    """Perceiver IO + TFT kept; graph memory replaced by a generic prompt."""
    c = _clone(cfg)
    c.graph.enabled = False
    c.llm.use_graph_prompt = False
    return c


def rag_llm(cfg: Config) -> Config:
    """Graph retrieval kept; decoding temperature fixed (fatigue ignored)."""
    c = _clone(cfg)
    c.tft.enabled = False
    # Fixed-temperature decoding: collapse the fatigue schedule to one value.
    mid = (cfg.llm.temp_low_fatigue + cfg.llm.temp_high_fatigue) / 2.0
    c.llm.temp_low_fatigue = mid
    c.llm.temp_high_fatigue = mid
    # Verbosity and option count no longer shrink with fatigue.
    c.llm.max_words_by_fatigue = {k: cfg.llm.max_words_by_fatigue["low"] for k in cfg.llm.max_words_by_fatigue}
    c.llm.candidates_by_fatigue = {k: cfg.llm.candidates_by_fatigue["low"] for k in cfg.llm.candidates_by_fatigue}
    return c


def ablate_tft(cfg: Config) -> Config:
    c = _clone(cfg)
    c.tft.enabled = False
    return c


def ablate_gat(cfg: Config) -> Config:
    c = _clone(cfg)
    c.graph.enabled = False
    c.llm.use_graph_prompt = False
    return c


def ablate_gate(cfg: Config) -> Config:
    c = _clone(cfg)
    c.safety.enabled = False
    return c


def ablate_cross_attention(cfg: Config) -> Config:
    c = _clone(cfg)
    c.perceiver.use_cross_attention = False
    return c


VARIANTS: Dict[str, Variant] = {
    "MCGA-LM": Variant("MCGA-LM", full_system),
    "LLM-Only": Variant("LLM-Only", llm_only, note="Sec. 4.3 baseline 2 (GPT-4o in the paper)"),
    "M-LLM": Variant("M-LLM", m_llm, note="Sec. 4.3 baseline 3; == \\GAT ablation (Table 7 footnote)"),
    "RAG-LLM": Variant("RAG-LLM", rag_llm, fixed_temperature=True, note="Sec. 4.3 baseline 4"),
    "\\PerceiverIO": Variant("\\PerceiverIO", full_system, ablate_perceiver=True, note="Sec. 4.4"),
    "\\TFT": Variant("\\TFT", ablate_tft, note="Sec. 4.4"),
    "\\GAT": Variant("\\GAT", ablate_gat, note="Sec. 4.4"),
    "\\BayesianGate": Variant("\\BayesianGate", ablate_gate, note="Sec. 4.4"),
    "\\cross-attention": Variant("\\cross-attention", ablate_cross_attention, note="Sec. 4.4"),
}


def get_variant(name: str) -> Variant:
    if name not in VARIANTS:
        raise KeyError(f"unknown variant {name!r}; available: {sorted(VARIANTS)}")
    return VARIANTS[name]


# --------------------------------------------------- full-factorial (D-10) -- #
# Sec. 4.4 calls its ablation "full-factorial" and then describes "removing one
# component at a time", which is a one-factor-at-a-time design: k+1 conditions,
# not 2^k. Fig. 4's five bars confirm the one-at-a-time reading, and that is what
# ``ABLATIONS`` and the default run implement. The genuinely full-factorial
# design the sentence claims is built below, because the two answer different
# questions -- one-at-a-time cannot detect an interaction, and the paper's own
# discussion ("the graph and adaptation modules remain the dominant
# contributors") is an interaction claim. See D-10 in docs/ASSUMPTIONS.md.

FACTORS: Dict[str, Callable[[Config], Config]] = {
    "PerceiverIO": full_system,  # handled by the runner's ablate_perceiver flag
    "TFT": ablate_tft,
    "GAT": ablate_gat,
    "BayesianGate": ablate_gate,
    "cross-attention": ablate_cross_attention,
}


def factorial_variant(removed: Sequence[str]) -> Variant:
    """A variant with every component in ``removed`` ablated simultaneously.

    ``removed=()`` is the full system; ``removed`` covering all five factors is
    the fully stripped system. Composing the transforms is sound because each
    one only sets configuration flags, and no two of them set the same flag to
    conflicting values.
    """
    unknown = [r for r in removed if r not in FACTORS]
    if unknown:
        raise KeyError(f"unknown factor(s) {unknown}; available: {sorted(FACTORS)}")
    ordered = [f for f in FACTORS if f in set(removed)]
    name = "MCGA-LM" if not ordered else "\\" + "+\\".join(ordered)

    def apply(cfg: Config) -> Config:
        c = _clone(cfg)
        for factor in ordered:
            c = FACTORS[factor](c)
        return c

    return Variant(
        name,
        apply,
        ablate_perceiver="PerceiverIO" in set(removed),
        note=f"Sec. 4.4 full-factorial cell: removed={list(ordered) or ['nothing']}",
    )


def full_factorial_design(factors: Sequence[str] = tuple(FACTORS)) -> List[Variant]:
    """All ``2^k`` cells of the design Sec. 4.4's first sentence claims.

    With the paper's five components this is 32 conditions rather than 6, which
    is why it is opt-in (``--full-factorial``) rather than the default: the cost
    is 32 training runs, and Fig. 4 reports the 6-condition version.
    """
    cells: List[Variant] = []
    for mask in range(1 << len(factors)):
        removed = [f for i, f in enumerate(factors) if mask & (1 << i)]
        cells.append(factorial_variant(removed))
    return cells
