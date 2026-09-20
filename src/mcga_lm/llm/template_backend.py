

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..data.taxonomy import PRAGMATIC_FUNCTIONS, SLOT_TYPE
from ..memory.graph import embed_lexeme
from .base import GeneratedUtterance
from .prompt import StructuredPrompt

# Surface realisations per pragmatic function, ordered long -> short so the
# fatigue-adaptive word budget (Sec. 3.5) picks a shorter form as fatigue rises.
TEMPLATES: Dict[str, Tuple[str, ...]] = {
    "request_object": ("Could you pass me the {slot} when you have a moment", "Please pass the {slot}", "{slot} please"),
    "request_action": ("Would you mind helping me with the {slot}", "Please help me {slot}", "Help {slot}"),
    "request_information": ("Can you tell me about the {slot}", "What about the {slot}", "{slot}?"),
    "request_drink": ("I would really like something to drink, some {slot} please", "I would like some {slot}", "{slot} please"),
    "request_food": ("I am getting hungry, could I have some {slot}", "I would like some {slot}", "{slot} please"),
    "request_medication": ("I think it is time for my {slot}", "I need my {slot}", "{slot} now"),
    "request_position_change": ("Could you help me adjust the {slot} a little", "Please move the {slot}", "Move {slot}"),
    "request_privacy": ("I would like some time on my own just now {slot}", "I need a moment alone {slot}", "Alone please"),
    "express_pain": ("The {slot} is worse than it was this morning", "My {slot} is bad", "{slot} hurts"),
    "express_discomfort": ("I am not comfortable with the {slot} like this", "The {slot} is uncomfortable", "{slot} wrong"),
    "express_fatigue": ("I am running out of energy, the {slot} has tired me", "I am tired after the {slot}", "Too tired"),
    "express_emotion": ("Thinking about the {slot} makes me feel a lot better", "The {slot} makes me happy", "Happy about {slot}"),
    "express_preference": ("I would much rather have the {slot} if that is alright", "I prefer the {slot}", "{slot} better"),
    "social_greeting": ("Hello {slot}, it is really good to see you today", "Hello {slot}, good to see you", "Hi {slot}"),
    "social_farewell": ("Goodbye {slot}, I will see you again soon", "Bye {slot}, see you soon", "Bye {slot}"),
    "social_thanks": ("Thank you {slot}, that made a real difference today", "Thank you {slot}", "Thanks {slot}"),
    "social_smalltalk": ("I was thinking about the {slot} earlier on today", "I was thinking about the {slot}", "About {slot}"),
    "ask_wellbeing": ("How have you been keeping since I saw you, {slot}", "How are you {slot}", "You okay {slot}"),
    "affirm": ("Yes, that is exactly what I meant about the {slot}", "Yes, the {slot}", "Yes"),
    "deny": ("No, that is not what I meant about the {slot}", "No, not the {slot}", "No"),
    "comment_observation": ("I noticed the {slot} has changed since yesterday", "Look at the {slot}", "The {slot}"),
    "narrate_past": ("I was remembering the {slot} from a long time ago", "I remember the {slot}", "Remember {slot}"),
    "plan_future": ("I would like to do the {slot} again later this week", "Let us do the {slot} later", "{slot} later"),
    "clarify": ("Let me put that another way, I meant the {slot}", "I meant the {slot}", "Mean {slot}"),
}

# Global "population frequency" lexicon: what a disembodied LLM would reach for
# in the absence of personal memory (Sec. 2: "trained on global frequency
# statistics alone"). Deliberately generic and persona-agnostic.
GLOBAL_LEXICON: Dict[str, Tuple[str, ...]] = {
    "Person": ("nurse", "doctor", "mum", "friend", "carer", "visitor", "neighbour"),
    "Object": ("water", "blanket", "window", "television", "phone", "pillow", "tea", "book"),
    "Activity": ("walk", "visit", "appointment", "meal", "shower", "call", "garden"),
    "AbstractState": ("pain", "comfort", "tiredness", "worry", "hope", "cold"),
}


class TemplateLanguageBackend:
    """Deterministic, weight-free stand-in for the paper's LLM (Sec. 3.5)."""

    def __init__(
        self,
        embedding_dim: int = 256,
        idiolect_bonus: float = 0.15,
        global_prior: float = 0.2,
        grounded_gain: float = 4.0,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.idiolect_bonus = idiolect_bonus
        # Retrieval biases generation without constraining it: global-lexicon
        # fillers stay in the candidate pool with a low prior even when a
        # sub-graph is supplied, so a hot decode can still wander off-graph --
        # which is how an LLM conditioned by RAG actually behaves. The
        # hallucination rate therefore falls out of the temperature schedule
        # (Sec. 3.5) rather than being set by hand. See A-30.
        self.global_prior = global_prior
        self.grounded_gain = grounded_gain

    # ------------------------------------------------------------------ #
    def generate(
        self,
        prompt: StructuredPrompt,
        n: int = 3,
        temperature: float = 1.0,
        top_p: float = 0.9,
        rng: Optional[np.random.Generator] = None,
    ) -> List[GeneratedUtterance]:
        rng = rng or np.random.default_rng()
        pool = self._candidate_pool(prompt)
        if not pool:
            return []
        scores = np.asarray([c.score for c in pool], dtype=float)
        # Temperature flattens the candidate distribution; top-p truncates it
        # (Sec. 3.5: top-p sampling with p = 0.9).
        probs = _softmax(scores / max(temperature, 1e-3))
        order = np.argsort(-probs)
        cumulative = np.cumsum(probs[order])
        keep = order[: max(1, int(np.searchsorted(cumulative, top_p) + 1))]
        k = min(n, len(keep))
        chosen = rng.choice(keep, size=k, replace=False, p=_renorm(probs[keep]))
        out: List[GeneratedUtterance] = []
        for idx in np.atleast_1d(chosen):
            cand = pool[int(idx)]
            out.append(self._realise(cand, prompt))
        return out

    # ------------------------------------------------------------------ #
    def _candidate_pool(self, prompt: StructuredPrompt) -> List["_Candidate"]:
        """Build (function, filler) candidates: retrieved ones plus global ones."""
        pool: List[_Candidate] = []
        if prompt.use_graph and prompt.active_nodes:
            for name, node_type, score in prompt.active_nodes:
                for fn in _functions_for_type(node_type):
                    bonus = self.idiolect_bonus if _in_idiolect(name, prompt) else 0.0
                    pool.append(
                        _Candidate(fn, name, float(score) * self.grounded_gain + bonus, grounded=True)
                    )
        # Global "population frequency" candidates are always available. Without
        # a sub-graph they are the only option (the LLM-Only / M-LLM baselines);
        # with one they sit far down the ranking and surface only when a high
        # decoding temperature flattens the distribution.
        for fn in PRAGMATIC_FUNCTIONS:
            slot_type = SLOT_TYPE[fn]
            for filler in GLOBAL_LEXICON[slot_type]:
                pool.append(_Candidate(fn, filler, self.global_prior, grounded=False))
        return pool

    def _realise(self, cand: "_Candidate", prompt: StructuredPrompt) -> GeneratedUtterance:
        """Pick the longest surface form that fits the fatigue word budget."""
        forms = TEMPLATES[cand.function]
        text = forms[-1].format(slot=cand.filler)
        for form in forms:
            candidate_text = form.format(slot=cand.filler)
            if len(candidate_text.split()) <= prompt.max_words:
                text = candidate_text
                break
        text = text.strip()
        text = text[0].upper() + text[1:] if text else text
        if not text.endswith((".", "?", "!")):
            text += "?" if cand.function in ("request_information", "ask_wellbeing") else "."
        return GeneratedUtterance(
            text=text,
            function=cand.function,
            source_nodes=(cand.filler,) if cand.grounded else (),
            entities=(cand.filler,),
            score=cand.score,
            grounded=cand.grounded,
        )

    # ------------------------------------------------------------------ #
    def embed_tokens(self, text: str, max_len: int = 24) -> np.ndarray:
        """Deterministic per-token embeddings for the intent-scoring head."""
        toks = text.lower().replace(".", "").replace("?", "").split()[:max_len]
        if not toks:
            return np.zeros((1, self.embedding_dim), dtype=np.float32)
        return np.stack([embed_lexeme(t, self.embedding_dim) for t in toks]).astype(np.float32)


class _Candidate:
    __slots__ = ("function", "filler", "score", "grounded")

    def __init__(self, function: str, filler: str, score: float, grounded: bool) -> None:
        self.function = function
        self.filler = filler
        self.score = score
        self.grounded = grounded


def _functions_for_type(node_type: str) -> Sequence[str]:
    return tuple(fn for fn in PRAGMATIC_FUNCTIONS if SLOT_TYPE[fn] == node_type)


def _in_idiolect(name: str, prompt: StructuredPrompt) -> bool:
    lowered = name.lower()
    return any(lowered in phrase.lower() for phrase in prompt.profile.idiolect)


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def _renorm(p: np.ndarray) -> np.ndarray:
    return p / p.sum()
