
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..config import SafetyConfig


@dataclass
class GateDecision:
    """Outcome of one gate evaluation."""

    accept: bool  # True -> present as a direct suggestion
    variance: float
    tau: float
    confidence: float = 0.0
    reason: str = ""


@dataclass
class ClarificationPrompt:
    """Clarification Mode display (Sec. 3.6): top-3 simplified Intent Bubbles."""

    bubbles: List[str]
    node_ids: List[int]


class BayesianGate:
    """Calibrated uncertainty gate with a user-configurable threshold."""

    def __init__(self, cfg: SafetyConfig, tau: Optional[float] = None) -> None:
        self.cfg = cfg
        self.tau = float(cfg.tau_default if tau is None else tau)

    # ------------------------------------------------------------------ #
    def decide(self, variance: float, confidence: float = 0.0) -> GateDecision:
        if not self.cfg.enabled:
            # Ablation "\ Bayesian Gate" (Sec. 4.4): present everything.
            return GateDecision(True, float(variance), self.tau, confidence, "gate disabled")
        accept = float(variance) < self.tau
        return GateDecision(
            accept=accept,
            variance=float(variance),
            tau=self.tau,
            confidence=float(confidence),
            reason="below threshold" if accept else "suppressed: uncertainty >= tau",
        )

    def tau_grid(self) -> np.ndarray:
        c = self.cfg
        n = int(round((c.tau_grid_stop - c.tau_grid_start) / c.tau_grid_step)) + 1
        return np.round(np.linspace(c.tau_grid_start, c.tau_grid_stop, n), 10)

    # ------------------------------------------------------------------ #
    def calibrate(
        self,
        variances: Sequence[float],
        accepted: Sequence[bool],
        rule: str = "largest",
    ) -> Tuple[float, List[dict]]:
        """Per-user threshold calibration (Sec. 3.6, "Calibration of the
        confidence threshold").

        ``variances``: Var(y_hat) for each calibration candidate.
        ``accepted``:  whether the simulated user accepted that candidate.
        Returns ``(tau, trace)`` where ``trace`` records FAR at each grid point.
        """
        v = np.asarray(variances, dtype=float)
        a = np.asarray(accepted, dtype=bool)
        trace: List[dict] = []
        feasible: List[float] = []
        for tau in self.tau_grid():
            passed = v < tau
            n_passed = int(passed.sum())
            far = float((passed & ~a).sum() / n_passed) if n_passed else 0.0
            trace.append({"tau": float(tau), "far": far, "n_passed": n_passed, "coverage": n_passed / max(len(v), 1)})
            if n_passed > 0 and far <= self.cfg.far_budget:
                feasible.append(float(tau))
        if not feasible:
            self.tau = float(self.cfg.tau_grid_start)
        elif rule == "smallest":  # literal reading of Sec. 3.6, see D-01
            self.tau = min(feasible)
        else:
            self.tau = max(feasible)
        return self.tau, trace


def clarification_bubbles(
    node_names: Sequence[str], node_ids: Sequence[int], k: int = 3
) -> ClarificationPrompt:
    """Build the top-3 Intent Bubbles shown in Clarification Mode (Sec. 3.6)."""
    labels = [_shorten(n) for n in node_names[:k]]
    return ClarificationPrompt(bubbles=labels, node_ids=[int(i) for i in node_ids[:k]])


def _shorten(name: str) -> str:
    """Bubbles are "simplified" one-word labels, e.g. "Pain", "Thirsty", "Bed"."""
    word = name.replace("_", " ").split()[0]
    return word[:1].upper() + word[1:]
