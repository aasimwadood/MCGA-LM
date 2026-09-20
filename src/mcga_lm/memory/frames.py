

from __future__ import annotations

import re
from typing import List, Optional, Protocol, Sequence

from .graph import RELATIONS, USER_NODE, IntentMemoryGraph, SemanticFrame

_TOKEN = re.compile(r"[a-z']+")

# Trigger words -> relation (ASSUMPTION A-05).
RELATION_TRIGGERS = {
    "interacts_with": ("tell", "ask", "talk", "call", "with", "thank", "greet"),
    "located_in": ("in", "at", "bedroom", "kitchen", "clinic", "garden", "outside"),
    "causes": ("because", "makes", "cause", "gives", "so"),
    "precedes": ("before", "then", "after", "later"),
    "expressed_as": ("feel", "hurt", "tired", "happy", "sad", "pain", "want", "need"),
    "associated_with": (),  # default
}

STATE_WORDS = {
    "pain",
    "tired",
    "fatigue",
    "thirsty",
    "hungry",
    "happy",
    "sad",
    "anxious",
    "comfortable",
    "uncomfortable",
    "cold",
    "hot",
    "breathless",
}

STOPWORDS = {
    "i",
    "a",
    "an",
    "the",
    "is",
    "am",
    "are",
    "to",
    "of",
    "my",
    "me",
    "you",
    "it",
    "and",
    "please",
    "would",
    "could",
    "like",
    "do",
    "does",
    "can",
    "this",
    "that",
    "for",
    "have",
    "has",
}


class FrameExtractor(Protocol):
    def __call__(self, utterance: str, graph: IntentMemoryGraph, partner: Optional[str] = None) -> SemanticFrame:
        ...


def tokenise(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class RuleBasedFrameExtractor:
    """Lexicon + graph-linking frame extractor (ASSUMPTION A-05)."""

    def __init__(self, min_token_len: int = 3) -> None:
        self.min_token_len = min_token_len

    def __call__(
        self, utterance: str, graph: IntentMemoryGraph, partner: Optional[str] = None
    ) -> SemanticFrame:
        tokens = tokenise(utterance)
        relation = self._relation(tokens)
        obj, obj_type = self._object(tokens, graph)
        intents = [t for t in tokens if t in STATE_WORDS]
        return SemanticFrame(
            subject=USER_NODE,
            subject_type="Person",
            predicate=relation,
            obj=obj,
            obj_type=obj_type,
            partner=partner,
            intent_nodes=tuple(dict.fromkeys(intents)),
        )

    def _relation(self, tokens: Sequence[str]) -> str:
        for relation in RELATIONS:
            triggers = RELATION_TRIGGERS.get(relation, ())
            if any(t in triggers for t in tokens):
                return relation
        return "associated_with"

    def _object(self, tokens: Sequence[str], graph: IntentMemoryGraph) -> tuple[str, str]:
        # Prefer a token that already names a graph node (entity linking).
        for t in tokens:
            if t in STOPWORDS:
                continue
            if graph.has_node(t):
                node = graph.nodes[graph.node_id(t)]
                return node.name, node.type
        for t in tokens:
            if t in STATE_WORDS:
                return t, "AbstractState"
        for t in tokens:
            if t not in STOPWORDS and len(t) >= self.min_token_len:
                return t, "Object"
        return "", "Object"
