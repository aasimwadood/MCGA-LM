

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .taxonomy import PRAGMATIC_FUNCTIONS, SLOT_TYPE

# Reference phrasings, deliberately *different wording* from the generation
# templates in llm/template_backend.py so BLEU/ROUGE are not trivially 1.0 when
# the system retrieves the right intent.
REFERENCE_TEMPLATES: Dict[str, Tuple[str, ...]] = {
    "request_object": ("can I have the {slot}", "I would like the {slot} now"),
    "request_action": ("give me a hand with the {slot}", "I need a hand with the {slot}"),
    "request_information": ("what is happening with the {slot}", "tell me about the {slot}"),
    "request_drink": ("I am thirsty, some {slot} would be good", "a drink of {slot} would help"),
    "request_food": ("I could manage a little {slot}", "some {slot} would be good now"),
    "request_medication": ("my {slot} is due about now", "I am ready for the {slot}"),
    "request_position_change": ("shift the {slot} up a bit for me", "the {slot} needs moving"),
    "request_privacy": ("give me a few minutes by myself {slot}", "some quiet time please {slot}"),
    "express_pain": ("the {slot} is hurting a lot today", "there is a lot of {slot} pain"),
    "express_discomfort": ("the {slot} does not feel right", "something is off with the {slot}"),
    "express_fatigue": ("the {slot} has worn me out", "I have no energy left after the {slot}"),
    "express_emotion": ("the {slot} lifted my mood", "I feel good about the {slot}"),
    "express_preference": ("the {slot} suits me better", "I would sooner have the {slot}"),
    "social_greeting": ("good to see you {slot}", "hello there {slot}"),
    "social_farewell": ("see you next time {slot}", "take care {slot}"),
    "social_thanks": ("that was kind of you {slot}", "I appreciate it {slot}"),
    "social_smalltalk": ("the {slot} came to mind today", "I keep thinking about the {slot}"),
    "ask_wellbeing": ("how is everything with you {slot}", "are you keeping well {slot}"),
    "affirm": ("that is right about the {slot}", "yes, the {slot}"),
    "deny": ("that is not the {slot} I meant", "no, not the {slot}"),
    "comment_observation": ("the {slot} looks different today", "something changed about the {slot}"),
    "narrate_past": ("the {slot} takes me back", "we used to do the {slot} often"),
    "plan_future": ("we should do the {slot} soon", "put the {slot} in for later"),
    "clarify": ("what I meant was the {slot}", "to be clear, the {slot}"),
}


@dataclass
class CorpusExample:
    function: str
    entity: str
    entity_type: str
    text: str


@dataclass
class AACIntentCorpus:
    """5,000 intent-utterance pairs, split 70/15/15 (Sec. 4.1)."""

    train: List[CorpusExample]
    val: List[CorpusExample]
    test: List[CorpusExample]

    @property
    def size(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)

    def all(self) -> List[CorpusExample]:
        return self.train + self.val + self.test


def realise_reference(function: str, entity: str, rng: Optional[np.random.Generator] = None) -> str:
    """Produce a reference utterance for BLEU-4 / ROUGE-L scoring (Sec. 4.2)."""
    forms = REFERENCE_TEMPLATES[function]
    idx = 0 if rng is None else int(rng.integers(len(forms)))
    text = forms[idx].format(slot=entity)
    return text[0].upper() + text[1:] + "."


def build_corpus(
    entity_pool: Dict[str, Sequence[str]],
    n_examples: int = 5000,
    seed: int = 0,
    splits: Tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> AACIntentCorpus:
    """Generate the synthetic AAC-Intent-Corpus (Sec. 4.1 shape)."""
    rng = np.random.default_rng(seed)
    examples: List[CorpusExample] = []
    for _ in range(n_examples):
        fn = str(rng.choice(PRAGMATIC_FUNCTIONS))
        slot_type = SLOT_TYPE[fn]
        pool = list(entity_pool.get(slot_type) or ["thing"])
        entity = str(rng.choice(pool))
        examples.append(
            CorpusExample(function=fn, entity=entity, entity_type=slot_type, text=realise_reference(fn, entity, rng))
        )
    rng.shuffle(examples)  # type: ignore[arg-type]
    n_train = int(splits[0] * n_examples)
    n_val = int(splits[1] * n_examples)
    return AACIntentCorpus(
        train=examples[:n_train],
        val=examples[n_train : n_train + n_val],
        test=examples[n_train + n_val :],
    )
