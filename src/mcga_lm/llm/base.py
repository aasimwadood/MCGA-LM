
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence

import numpy as np

from .prompt import StructuredPrompt


@dataclass
class GeneratedUtterance:
    """One decoded candidate."""

    text: str
    function: str = ""  # pragmatic function realised (taxonomy of Sec. 4.1)
    source_nodes: Sequence[str] = field(default_factory=tuple)  # graph nodes used
    entities: Sequence[str] = field(default_factory=tuple)  # entities named in the text
    score: float = 0.0  # backend's own ranking score
    grounded: bool = True  # False if a slot was filled from global statistics

    @property
    def tokens(self) -> List[str]:
        return self.text.replace(",", " ").replace(".", " ").split()

    @property
    def n_words(self) -> int:
        return len(self.tokens)


class LanguageBackend(Protocol):
    """Minimal contract used by :class:`~mcga_lm.pipeline.MCGALM`."""

    embedding_dim: int

    def generate(
        self,
        prompt: StructuredPrompt,
        n: int = 3,
        temperature: float = 1.0,
        top_p: float = 0.9,
        rng: Optional[np.random.Generator] = None,
    ) -> List[GeneratedUtterance]:
        """Decode ``n`` candidate utterances (once -- Sec. 3.6 decodes once)."""
        ...

    def embed_tokens(self, text: str, max_len: int = 24) -> np.ndarray:
        """(L, embedding_dim) token embeddings fed to the intent-scoring head."""
        ...
