

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple

from ..data.lexicons import FIRST_NAMES, ROLES
from ..memory.graph import IntentMemoryGraph

# Object nodes whose misuse is clinically consequential (Sec. 2's pain-medication
# example). A drug named for a user whose graph does not carry it is treated as
# a contradiction, not a benign addition.
MEDICATION_TERMS: Set[str] = {
    "baclofen",
    "gabapentin",
    "paracetamol",
    "morphine",
    "riluzole",
    "medication",
    "painkiller",
    "sedative",
    "antibiotic",
}

PERSON_TERMS: Set[str] = {n.lower() for n in FIRST_NAMES} | {r.lower() for r in ROLES} | {
    "nurse",
    "doctor",
    "mum",
    "dad",
    "carer",
    "visitor",
    "neighbour",
    "consultant",
    "physio",
}


class HallucinationKind(str, Enum):
    NONE = "none"
    SOFT = "soft"
    HARD = "hard"


@dataclass
class HallucinationVerdict:
    kind: HallucinationKind
    entity: str = ""
    reason: str = ""

    @property
    def is_hard(self) -> bool:
        return self.kind is HallucinationKind.HARD

    @property
    def is_soft(self) -> bool:
        return self.kind is HallucinationKind.SOFT


class HallucinationDetector:
    """Rule-based entity linking against ``G`` and the dialogue history (Sec. 4.2)."""

    def __init__(self, safety_critical: Optional[Set[str]] = None) -> None:
        self.medications = set(safety_critical or MEDICATION_TERMS)

    # ------------------------------------------------------------------ #
    def classify(
        self,
        entities: Sequence[str],
        graph: IntentMemoryGraph,
        history: Iterable[Tuple[str, str]] = (),
        known_partners: Sequence[str] = (),
    ) -> HallucinationVerdict:
        """Classify one utterance from the entities it names.

        Precedence: a hard contradiction anywhere outranks a soft addition.
        """
        history_tokens = _history_tokens(history)
        partner_names = {p.lower() for p in known_partners}
        soft: Optional[HallucinationVerdict] = None

        for raw in entities:
            entity = raw.strip().lower()
            if not entity:
                continue
            if graph.has_node(entity) or entity in history_tokens:
                continue  # licensed by the user's own memory or the conversation

            # (a) a person the user's relational world does not contain: the
            #     canonical hard case, "naming the wrong caregiver" (Sec. 4.2).
            if entity in PERSON_TERMS and entity not in partner_names:
                return HallucinationVerdict(
                    HallucinationKind.HARD, raw, "names a person absent from the intent graph"
                )
            # (b) a medication the user's graph does not carry (Sec. 2).
            if entity in self.medications:
                return HallucinationVerdict(
                    HallucinationKind.HARD, raw, "names a medication absent from the intent graph"
                )
            # (c) otherwise: present but not contradicted -> soft.
            if soft is None:
                soft = HallucinationVerdict(
                    HallucinationKind.SOFT, raw, "entity absent from graph and history, not contradicted"
                )

        if soft is not None:
            return soft
        return HallucinationVerdict(HallucinationKind.NONE)

    # ------------------------------------------------------------------ #
    def relation_contradiction(
        self, subject: str, relation: str, obj: str, graph: IntentMemoryGraph
    ) -> Optional[HallucinationVerdict]:
        """Hard case (b): a relation that conflicts with an existing edge.

        ``located_in`` is treated as functional -- an object lives in one place --
        so asserting a different place contradicts the graph (ASSUMPTION A-24).
        """
        if relation != "located_in" or not graph.has_node(subject):
            return None
        sid = graph.node_id(subject)
        for e in graph.edges:
            if e.src == sid and e.relation == "located_in":
                existing = graph.nodes[e.dst].name.lower()
                if existing != obj.strip().lower():
                    return HallucinationVerdict(
                        HallucinationKind.HARD,
                        obj,
                        f"asserts located_in '{obj}' but graph has '{existing}'",
                    )
        return None


def _history_tokens(history: Iterable[Tuple[str, str]]) -> Set[str]:
    tokens: Set[str] = set()
    for speaker, text in history:
        tokens.add(speaker.lower())
        tokens.update(text.lower().replace(".", "").replace("?", "").split())
    return tokens


def summarise(verdicts: Sequence[HallucinationVerdict]) -> Dict[str, float]:
    """Hard rate (``H``) and soft rate, reported separately as in Sec. 4.2."""
    n = len(verdicts)
    if n == 0:
        return {"hard_rate": float("nan"), "soft_rate": float("nan"), "n": 0}
    hard = sum(1 for v in verdicts if v.is_hard)
    soft = sum(1 for v in verdicts if v.is_soft)
    return {"hard_rate": hard / n, "soft_rate": soft / n, "n": n}
