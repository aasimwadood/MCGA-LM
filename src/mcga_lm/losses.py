

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .config import LossConfig


@dataclass
class LossBreakdown:
    total: torch.Tensor
    fatigue: torch.Tensor
    contrastive: torch.Tensor
    recon: torch.Tensor
    intent: torch.Tensor

    def as_floats(self) -> Dict[str, float]:
        return {
            "total": float(self.total.detach()),
            "fatigue": float(self.fatigue.detach()),
            "contrastive": float(self.contrastive.detach()),
            "recon": float(self.recon.detach()),
            "intent": float(self.intent.detach()),
        }


def fatigue_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
  
    return F.mse_loss(predicted.reshape(-1), target.reshape(-1))


def contrastive_loss(
    z_ctx: torch.Tensor, z_utt: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """Eq. (11): InfoNCE with in-batch negatives ``N``.

    z_ctx, z_utt: (B, D). Cosine similarity, temperature tau_c.
    """
    if z_ctx.shape[0] < 2:
        return z_ctx.new_zeros(())
    a = F.normalize(z_ctx, dim=-1)
    b = F.normalize(z_utt, dim=-1)
    logits = (a @ b.t()) / temperature  # (B, B); diagonal is the positive pair
    labels = torch.arange(a.shape[0], device=a.device)
    return F.cross_entropy(logits, labels)


def reconstruction_loss(x_phys: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """Eq. (12): mean squared error over the physiological window."""
    return F.mse_loss(reconstructed, x_phys)


def intent_loss(node_scores: torch.Tensor, targets: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Eq. (13): cross-entropy of the GAT's normalised attention against the
    binary ground-truth intent nodes.

    node_scores: (B, N) normalised attention (y_hat); targets: (B, N) in {0, 1}.
    The sum in Eq. (13) is over nodes; we average over the batch.
    """
    log_scores = torch.log(node_scores.clamp_min(eps))
    per_example = -(targets * log_scores).sum(dim=-1)
    # Normalise by the number of positive nodes so examples with differently
    # sized intent sets contribute comparably (ASSUMPTION A-13).
    n_pos = targets.sum(dim=-1).clamp_min(1.0)
    return (per_example / n_pos).mean()


def total_loss(
    cfg: LossConfig,
    fatigue_pred: Optional[torch.Tensor] = None,
    fatigue_target: Optional[torch.Tensor] = None,
    z_ctx: Optional[torch.Tensor] = None,
    z_utt: Optional[torch.Tensor] = None,
    x_phys: Optional[torch.Tensor] = None,
    x_recon: Optional[torch.Tensor] = None,
    node_scores: Optional[torch.Tensor] = None,
    intent_targets: Optional[torch.Tensor] = None,
    device=None,
) -> LossBreakdown:
    """Assemble Eq. (9). Any term whose inputs are ``None`` contributes zero,
    which is how the paper runs the synthetic stage without ``L_fatigue``."""
    zero = torch.zeros((), device=device)
    l_fat = fatigue_loss(fatigue_pred, fatigue_target) if fatigue_pred is not None and fatigue_target is not None else zero
    l_con = contrastive_loss(z_ctx, z_utt, cfg.contrastive_temp) if z_ctx is not None and z_utt is not None else zero
    l_rec = reconstruction_loss(x_phys, x_recon) if x_phys is not None and x_recon is not None else zero
    l_int = intent_loss(node_scores, intent_targets) if node_scores is not None and intent_targets is not None else zero
    total = l_fat + cfg.lambda_contrastive * l_con + cfg.lambda_recon * l_rec + cfg.lambda_intent * l_int
    return LossBreakdown(total=total, fatigue=l_fat, contrastive=l_con, recon=l_rec, intent=l_int)
