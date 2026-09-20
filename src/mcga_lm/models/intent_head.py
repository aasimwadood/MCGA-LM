

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class UncertaintyEstimate:
    """Result of the MC-Dropout pass bank (Eq. 8)."""

    variance: torch.Tensor  # (B,) predictive variance Var(y_hat)
    confidence: torch.Tensor  # (B,) mean token probability across passes
    per_token_mean: torch.Tensor  # (B, L)
    per_token_var: torch.Tensor  # (B, L)
    n_passes: int


class IntentScoringHead(nn.Module):
    """Scores an already-decoded utterance against the multimodal context."""

    def __init__(self, context_dim: int, token_dim: int, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),  # kept active at inference for MC Dropout
        )
        self.token_proj = nn.Sequential(
            nn.Linear(token_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.score = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.dropout_p = dropout

    def forward(self, context: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """context: (B, C); tokens: (B, L, E) -> per-token probability (B, L)."""
        c = self.context_proj(context).unsqueeze(1).expand(-1, tokens.shape[1], -1)
        t = self.token_proj(tokens)
        logits = self.score(torch.cat([c, t], dim=-1)).squeeze(-1)
        return torch.sigmoid(logits)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def mc_dropout(
        self,
        context: torch.Tensor,
        tokens: torch.Tensor,
        n_passes: int = 20,
        mask: Optional[torch.Tensor] = None,
    ) -> UncertaintyEstimate:
        """``N``-pass MC Dropout re-scoring (paper Sec. 3.6, Eq. 8).

        ``mask`` (B, L) marks real tokens so padding does not enter the mean.
        """
        was_training = self.training
        self.train(True)  # keep dropout active at inference (Gal & Ghahramani, 2016)
        try:
            passes = torch.stack([self.forward(context, tokens) for _ in range(n_passes)], dim=0)
        finally:
            self.train(was_training)

        per_token_mean = passes.mean(dim=0)  # (B, L)
        # Unbiased variance across the N sampled weight sets w^(n).
        per_token_var = passes.var(dim=0, unbiased=True) if n_passes > 1 else torch.zeros_like(per_token_mean)

        if mask is None:
            variance = per_token_var.mean(dim=-1)
            confidence = per_token_mean.mean(dim=-1)
        else:
            m = mask.to(per_token_var.dtype)
            denom = m.sum(dim=-1).clamp_min(1.0)
            variance = (per_token_var * m).sum(dim=-1) / denom
            confidence = (per_token_mean * m).sum(dim=-1) / denom
        return UncertaintyEstimate(
            variance=variance,
            confidence=confidence,
            per_token_mean=per_token_mean,
            per_token_var=per_token_var,
            n_passes=n_passes,
        )


class PhysiologyDecoder(nn.Module):
    """Reconstruction decoder for ``L_recon`` (paper Eq. 12, Sec. 3.10).

    "The decoder is a lightweight 3-layer MLP with 256 hidden units."
    """

    def __init__(self, latent_dim: int, out_steps: int, out_features: int, hidden: int = 256, layers: int = 3) -> None:
        super().__init__()
        self.out_steps = out_steps
        self.out_features = out_features
        blocks: list[nn.Module] = []
        dim = latent_dim
        for _ in range(layers - 1):
            blocks += [nn.Linear(dim, hidden), nn.GELU()]
            dim = hidden
        blocks.append(nn.Linear(dim, out_steps * out_features))
        self.net = nn.Sequential(*blocks)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """latents: (B, M, D) or (B, D) -> reconstructed x_phys (B, T_p, d_p)."""
        if latents.dim() == 3:
            latents = latents.mean(dim=1)
        out = self.net(latents)
        return out.view(-1, self.out_steps, self.out_features)
