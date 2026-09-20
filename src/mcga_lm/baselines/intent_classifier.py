

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from ..config import Config
from ..data.corpus import realise_reference
from ..data.personas import Persona
from ..data.taxonomy import PRAGMATIC_FUNCTIONS, SLOT_TYPE
from ..models.perceiver_io import MultimodalBatch, build_context_encoder
from ..models.tft import TemporalFusionTransformer
from ..data.dataset import sliding_windows


class IntentClassifierBaseline(nn.Module):
    """Front end of MCGA-LM + a feed-forward classifier over 24 functions."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.context_encoder = build_context_encoder(cfg.inputs, cfg.perceiver, ablate_perceiver=False)
        self.tft = TemporalFusionTransformer(self.context_encoder.output_dim, cfg.tft)
        in_dim = self.context_encoder.output_dim + cfg.tft.state_dim
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, len(PRAGMATIC_FUNCTIONS)),
        )

    def forward(self, batch: MultimodalBatch) -> torch.Tensor:
        z = self.context_encoder(batch)
        state = self.tft(sliding_windows(z, self.cfg.tft.window))
        return self.classifier(torch.cat([z, state.vector], dim=-1))

    @torch.no_grad()
    def predict(self, batch: MultimodalBatch, top_k: int = 5) -> torch.Tensor:
        """Ranked function indices per turn, for IHR@K."""
        return self.forward(batch).topk(k=min(top_k, len(PRAGMATIC_FUNCTIONS)), dim=-1).indices


@dataclass
class StoredPhrase:
    function: str
    entity: str
    text: str
    weight: float


class PhraseStore:
    """The persona's stored phrase inventory -- one phrase per function.

    Built from the persona's own graph weights, so "highest-weighted" means the
    entity most strongly associated with the user for that function's slot type.
    """

    def __init__(self, persona: Persona) -> None:
        self.entries: Dict[str, StoredPhrase] = {}
        rng = np.random.default_rng(persona.seed)
        for fn in PRAGMATIC_FUNCTIONS:
            pool = persona.nodes_of_type(SLOT_TYPE[fn])
            if not pool:
                continue
            best = max(pool, key=lambda name: persona.graph.neighbour_weight("User", name))
            self.entries[fn] = StoredPhrase(
                function=fn,
                entity=best,
                text=realise_reference(fn, best, rng),
                weight=persona.graph.neighbour_weight("User", best),
            )

    def retrieve(self, function: str) -> Optional[StoredPhrase]:
        return self.entries.get(function)

    def __len__(self) -> int:
        return len(self.entries)
