
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..config import InputDims
from ..memory.graph import IntentMemoryGraph, embed_lexeme
from ..models.perceiver_io import MultimodalBatch
from .personas import Persona, PersonaTurn
from .physiology import PhysiologySynthesiser, environment_vector


@dataclass
class SessionTensors:
    """One simulated session, ready for the encoder."""

    batch: MultimodalBatch  # each field has leading dim = n_turns
    fatigue: torch.Tensor  # (N,) ground-truth fatigue index (Sec. 4.5)
    intent_targets: torch.Tensor  # (N, |V|) binary node indicators for Eq. (13)
    utterance_embeddings: torch.Tensor  # (N, D_w) pooled reference utterance, Eq. (11)
    turns: List[PersonaTurn]

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    def to(self, device) -> "SessionTensors":
        return SessionTensors(
            batch=self.batch.to(device),
            fatigue=self.fatigue.to(device),
            intent_targets=self.intent_targets.to(device),
            utterance_embeddings=self.utterance_embeddings.to(device),
            turns=self.turns,
        )


class TurnEncoder:
    """Materialises ``X_t`` (Eq. 2) for simulated turns."""

    def __init__(
        self,
        dims: InputDims,
        reduced_sensor_set: bool = False,
        switchboard=None,
    ) -> None:
        self.dims = dims
        # Sec. 3.9: sensing is under the user's control, so the switchboard
        # travels with the encoder that materialises X_t.
        self.physio = PhysiologySynthesiser(
            dims, reduced_sensor_set=reduced_sensor_set, switchboard=switchboard
        )
        self.switchboard = self.physio.switchboard

    # ------------------------------------------------------------------ #
    def encode_turn(self, turn: PersonaTurn, rng: np.random.Generator) -> dict:
        d = self.dims
        phys = self.physio.physiology(turn.context.fatigue, turn.context.arousal, rng)
        beh = self.physio.behaviour(turn.context.fatigue, turn.context.arousal, rng)
        env = environment_vector(
            d,
            turn.context.location_id,
            turn.context.partner_id,
            turn.context.noise_db,
            turn.context.time_of_day,
        )
        env = env * self.switchboard.env_mask(d)  # Sec. 3.9
        ling = self.encode_history(turn.history)
        return {"phys": phys, "beh": beh, "env": env, "ling": ling}

    def encode_history(self, history: Sequence[Tuple[str, str]]) -> np.ndarray:
        """``x_ling``: the last L = 10 tokens of dialogue history (Sec. 3.2)."""
        d = self.dims
        tokens: List[str] = []
        for speaker, text in history:
            tokens.extend([speaker.lower()] + text.lower().replace(".", "").split())
        tokens = tokens[-d.ling_tokens :]
        out = np.zeros((d.ling_tokens, d.ling_dim), dtype=np.float32)
        for i, tok in enumerate(tokens):
            out[d.ling_tokens - len(tokens) + i] = embed_lexeme(tok, d.ling_dim)
        return out

    # ------------------------------------------------------------------ #
    def encode_session(
        self,
        persona: Persona,
        turns: Sequence[PersonaTurn],
        graph: Optional[IntentMemoryGraph] = None,
        seed: int = 0,
    ) -> SessionTensors:
        rng = np.random.default_rng(seed)
        graph = graph if graph is not None else persona.graph
        parts = [self.encode_turn(t, rng) for t in turns]
        batch = MultimodalBatch(
            phys=torch.from_numpy(np.stack([p["phys"] for p in parts])),
            beh=torch.from_numpy(np.stack([p["beh"] for p in parts])),
            env=torch.from_numpy(np.stack([p["env"] for p in parts])),
            ling=torch.from_numpy(np.stack([p["ling"] for p in parts])),
        )
        fatigue = torch.tensor([t.context.fatigue for t in turns], dtype=torch.float32)
        targets = torch.zeros(len(turns), len(graph.nodes), dtype=torch.float32)
        for i, t in enumerate(turns):
            for name in t.intent.entities:
                if graph.has_node(name):
                    targets[i, graph.node_id(name)] = 1.0
        utt = torch.from_numpy(
            np.stack([_pool_utterance(t.intent.reference, self.dims.ling_dim) for t in turns])
        )
        return SessionTensors(
            batch=batch,
            fatigue=fatigue,
            intent_targets=targets,
            utterance_embeddings=utt,
            turns=list(turns),
        )


def _pool_utterance(text: str, dim: int) -> np.ndarray:
    """Mean-pooled embedding of an accepted utterance (``z_utt`` of Eq. 11)."""
    toks = text.lower().replace(".", "").replace("?", "").split()
    if not toks:
        return np.zeros(dim, dtype=np.float32)
    return np.stack([embed_lexeme(t, dim) for t in toks]).mean(axis=0).astype(np.float32)


def sliding_windows(sequence: torch.Tensor, window: int) -> torch.Tensor:
    """Left-padded sliding windows for the TFT (Sec. 3.3, W = 32).

    sequence: (N, D) -> (N, W, D), where window ``i`` ends at turn ``i`` and
    earlier positions repeat the first available embedding when the session has
    not yet produced ``W`` turns.
    """
    n, d = sequence.shape
    out = sequence.new_zeros(n, window, d)
    for i in range(n):
        start = i - window + 1
        if start >= 0:
            out[i] = sequence[start : i + 1]
        else:
            pad = -start
            out[i, :pad] = sequence[0].unsqueeze(0).expand(pad, d)
            out[i, pad:] = sequence[: i + 1]
    return out
