

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..config import TFTConfig
# Re-exported so ``from mcga_lm.models.tft import fatigue_level`` keeps working.
from ..states import FATIGUE_BOUNDARIES, FATIGUE_LEVELS, fatigue_level  # noqa: F401
from .layers import GatedResidualNetwork


@dataclass
class CognitiveState:
    """Output of Phase II."""

    vector: torch.Tensor  # s_cog,t (B, D_c)
    fatigue: torch.Tensor  # scalar fatigue index in [0, 1] (B,)
    cognitive_load: torch.Tensor  # (B,)
    arousal: torch.Tensor  # (B,)
    selection_weights: Optional[torch.Tensor] = None  # v_t (B, D), Eq. (5)

    def level(self, index: int = 0) -> str:
        """Discretise fatigue into low/moderate/high (Sec. 3.5 component 2)."""
        return fatigue_level(float(self.fatigue[index]))


class VariableSelectionNetwork(nn.Module):
    """Sparse per-dimension weighting ``v_t^{(d)}`` (paper Eq. 5)."""

    def __init__(self, dim: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.grn = GatedResidualNetwork(dim, hidden, output_dim=dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, W, D) -> (weighted x, softmax weights over D)."""
        weights = torch.softmax(self.grn(x), dim=-1)
        # Rescale by D so that a uniform selection is the identity and the LSTM
        # input keeps its scale (ASSUMPTION; softmax alone shrinks by 1/D).
        return x * weights * x.shape[-1], weights


class TemporalFusionTransformer(nn.Module):
    """TFT cognitive-state estimator (paper Sec. 3.3)."""

    def __init__(self, input_dim: int, cfg: TFTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vsn = VariableSelectionNetwork(input_dim, cfg.lstm_hidden, cfg.dropout)
        self.lstm = nn.LSTM(input_dim, cfg.lstm_hidden, batch_first=True)
        self.gate = GatedResidualNetwork(cfg.lstm_hidden, cfg.lstm_hidden, dropout=cfg.dropout)
        self.attn = nn.MultiheadAttention(
            cfg.lstm_hidden, cfg.attn_heads, dropout=cfg.dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(cfg.lstm_hidden)
        self.state_proj = nn.Sequential(
            nn.Linear(cfg.lstm_hidden, cfg.state_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.state_dim),
        )
        # Three interpretable scalars read off s_cog (Sec. 3.3); index 0 is the
        # fatigue score supervised by L_fatigue (Eq. 10).
        self.state_head = nn.Linear(cfg.state_dim, 3)

    def forward(self, sequence: torch.Tensor) -> CognitiveState:
        """sequence: (B, W, D) of contextual embeddings Z_ctx,t-W+1:t."""
        if not self.cfg.enabled:
            return self._clamped_low_fatigue(sequence)

        selected, weights = self.vsn(sequence)  # Eq. (5)
        encoded, _ = self.lstm(selected)
        encoded = self.gate(encoded)
        attended, _ = self.attn(encoded, encoded, encoded, need_weights=False)
        encoded = self.attn_norm(encoded + attended)
        final = encoded[:, -1, :]  # "the TFT's final-step output" (Sec. 3.3)
        state = self.state_proj(final)
        scalars = torch.sigmoid(self.state_head(state))
        return CognitiveState(
            vector=state,
            fatigue=scalars[:, 0],
            cognitive_load=scalars[:, 1],
            arousal=scalars[:, 2],
            selection_weights=weights[:, -1, :],
        )

    def _clamped_low_fatigue(self, sequence: torch.Tensor) -> CognitiveState:
        """Ablation "\\ TFT" (Sec. 4.4): s_cog fixed to the low-fatigue state."""
        b = sequence.shape[0]
        device, dtype = sequence.device, sequence.dtype
        zeros = torch.zeros(b, device=device, dtype=dtype)
        return CognitiveState(
            vector=torch.zeros(b, self.cfg.state_dim, device=device, dtype=dtype),
            fatigue=zeros,
            cognitive_load=zeros,
            arousal=zeros,
            selection_weights=None,
        )

    @property
    def output_dim(self) -> int:
        return self.cfg.state_dim
