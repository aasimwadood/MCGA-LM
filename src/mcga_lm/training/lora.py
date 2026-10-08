"""LoRA personalisation of the language model (paper Sec. 3.5, 3.6, 3.7, 4.9).

What the paper specifies, and where it lives here:

  * Sec. 3.5: adapters (r = 16, alpha = 32) "are updated, on the user's accepted
    utterances paired with their prompts". :func:`personalise` trains them on
    (prompt, accepted utterance) pairs, one adapter per user.
  * Sec. 3.6: "Accepted utterances update the intent graph and, after a 24-hour
    on-device queue, fine-tune the LoRA adapters." :class:`LoRAUpdateQueue`
    holds accepted pairs until they are 24 hours old.
  * Sec. 4.9: "LoRA fine-tuning converges in 3 epochs per user"
    (``TrainingConfig.lora_epochs``).
  * Sec. 3.7 stage 3: "the LLM (instruction-tuned on a generic AAC prompt
    dataset) is then fine-tuned with LoRA". :func:`instruction_tune` runs that
    step on a dataset the caller supplies; the dataset itself is not published
    with the paper, so nothing here can stand in for it.

Sec. 3.5 also calls the update scheme "federated so model updates never leave
the device". Training here is local to the process holding the model, which
satisfies the second half. The paper describes no aggregation step, server or
schedule, so there is nothing further to implement for the first half.

The template backend has no weights, so these functions report a skip for it
rather than pretending to train.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..utils import get_logger

logger = get_logger(__name__)

Pair = Tuple[str, str]  # (rendered prompt, accepted utterance)


@dataclass
class QueuedUtterance:
    prompt: str
    utterance: str
    accepted_at_h: float  # simulation clock, hours


class LoRAUpdateQueue:
    """The 24-hour on-device queue of Sec. 3.6.

    Accepted (prompt, utterance) pairs wait here until they are ``hours`` old,
    then :meth:`release` hands them to :func:`personalise`.
    """

    def __init__(self, hours: float = 24.0) -> None:
        self.hours = float(hours)
        self._items: List[QueuedUtterance] = []

    def add(self, prompt: str, utterance: str, now_h: float) -> None:
        self._items.append(QueuedUtterance(prompt, utterance, float(now_h)))

    def release(self, now_h: float) -> List[Pair]:
        """Remove and return every pair that has waited at least ``hours``."""
        due = [q for q in self._items if now_h - q.accepted_at_h >= self.hours]
        self._items = [q for q in self._items if now_h - q.accepted_at_h < self.hours]
        return [(q.prompt, q.utterance) for q in due]

    def __len__(self) -> int:
        return len(self._items)


def supports_lora(backend) -> bool:
    return bool(getattr(backend, "supports_lora", False))


def personalise(
    backend,
    pairs: Sequence[Pair],
    adapter: str,
    epochs: int,
    lr: float,
    reset: bool = True,
) -> Dict[str, object]:
    """Fit one user's LoRA adapter on their accepted utterances (Sec. 3.5).

    ``reset=True`` starts the adapter from the shared (instruction-tuned) one, so
    retraining a user does not stack on a previous system's run. The flush of
    :class:`LoRAUpdateQueue` passes ``reset=False`` to continue training.
    """
    if not supports_lora(backend):
        return {"adapter": adapter, "skipped": "backend has no LoRA adapters", "n_pairs": len(pairs)}
    if not pairs:
        return {"adapter": adapter, "skipped": "no accepted utterances", "n_pairs": 0}
    backend.use_adapter(adapter, create=True, reset=reset)
    history = backend.fine_tune(pairs, epochs=epochs, lr=lr)
    return {"adapter": adapter, "n_pairs": len(pairs), "epochs": epochs, "history": history}


def flush_queue(
    backend,
    queue: LoRAUpdateQueue,
    now_h: float,
    adapter: str,
    epochs: int,
    lr: float,
) -> Dict[str, object]:
    """Train on whatever the 24-hour queue releases at ``now_h``."""
    pairs = queue.release(now_h)
    return personalise(backend, pairs, adapter, epochs=epochs, lr=lr, reset=False)


def load_instruction_pairs(path: str | Path) -> List[Pair]:
    """Read a JSON-lines file of ``{"prompt": ..., "response": ...}`` records."""
    pairs: List[Pair] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "prompt" not in record or "response" not in record:
                raise ValueError(f"{path}:{line_no}: expected keys 'prompt' and 'response'")
            pairs.append((str(record["prompt"]), str(record["response"])))
    return pairs


def instruction_tune(
    backend, pairs: Optional[Sequence[Pair]], epochs: int, lr: float
) -> Dict[str, object]:
    """Sec. 3.7 stage 3: instruction-tune the shared adapter before personalisation.

    Per-user adapters are created as copies of this one.
    """
    if not supports_lora(backend):
        return {"skipped": "backend has no LoRA adapters"}
    if not pairs:
        logger.warning(
            "Sec. 3.7 instruction-tunes the LLM on 'a generic AAC prompt dataset' before "
            "personalisation. That dataset is not published, so this step was skipped; pass "
            "--instruction-data <file.jsonl> to run it."
        )
        return {"skipped": "no instruction dataset supplied"}
    backend.use_adapter(backend.base_adapter, create=False)
    history = backend.fine_tune(pairs, epochs=epochs, lr=lr)
    return {"n_pairs": len(pairs), "epochs": epochs, "history": history}
