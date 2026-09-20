
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..config import SafetyConfig
from ..utils import get_logger

_logger = get_logger("mcga_lm.safety")


class FARBudgetUnreachable(RuntimeError):
    """No (tau, floor) setting keeps the false-acceptance rate within budget.

    Raised only when ``SafetyConfig.strict_far_budget`` is set. The condition
    itself is always recorded on ``BayesianGate.budget_met``.
    """


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

    def __init__(
        self,
        cfg: SafetyConfig,
        tau: Optional[float] = None,
        confidence_floor: Optional[float] = None,
    ) -> None:
        self.cfg = cfg
        self.tau = float(cfg.tau_default if tau is None else tau)
        # Confidence floor of the FAR guard. 0.0 reproduces Sec. 3.6's rule.
        self.confidence_floor = float(
            getattr(cfg, "confidence_floor", 0.0) if confidence_floor is None else confidence_floor
        )
        # Set by calibrate(): whether the FAR budget was actually reachable.
        # None means "not calibrated yet", so an uncalibrated gate is not
        # mistaken for one that passed its budget.
        self.budget_met: Optional[bool] = None

    @property
    def guarded(self) -> bool:
        """Whether the confidence floor participates in the decision."""
        return getattr(self.cfg, "decision_rule", "variance_only") == "guarded"

    # ------------------------------------------------------------------ #
    def decide(self, variance: float, confidence: float = 0.0) -> GateDecision:
        """Sec. 3.6's decision rule, plus the FAR guard when enabled.

        Printed rule: accept iff ``Var(y_hat) < tau``. Under ``decision_rule =
        'guarded'`` the candidate must also clear the calibrated confidence
        floor, because variance alone cannot separate "the samples agree and
        are right" from "the samples agree and are wrong".
        """
        if not self.cfg.enabled:
            # Ablation "\ Bayesian Gate" (Sec. 4.4): present everything.
            return GateDecision(True, float(variance), self.tau, confidence, "gate disabled")

        v, c = float(variance), float(confidence)
        below_tau = v < self.tau
        floor = self.confidence_floor if self.guarded else 0.0
        clears_floor = c >= floor

        if not below_tau:
            reason = "suppressed: uncertainty >= tau"
        elif not clears_floor:
            reason = f"suppressed: confidence {c:.3f} < floor {floor:.2f}"
        else:
            reason = "below threshold"
        return GateDecision(
            accept=bool(below_tau and clears_floor),
            variance=v,
            tau=self.tau,
            confidence=c,
            reason=reason,
        )

    def tau_grid(self) -> np.ndarray:
        c = self.cfg
        n = int(round((c.tau_grid_stop - c.tau_grid_start) / c.tau_grid_step)) + 1
        return np.round(np.linspace(c.tau_grid_start, c.tau_grid_stop, n), 10)

    def floor_grid(self) -> np.ndarray:
        """Candidate confidence floors. 0.0 first, so the printed rule is tried
        before any deviation from it is considered."""
        c = self.cfg
        stop = float(getattr(c, "floor_grid_stop", 0.9))
        step = float(getattr(c, "floor_grid_step", 0.1))
        n = int(round(stop / step)) + 1
        return np.round(np.linspace(0.0, stop, n), 10)

    # ------------------------------------------------------------------ #
    def calibrate(
        self,
        variances: Sequence[float],
        accepted: Sequence[bool],
        rule: str = "largest",
        confidences: Optional[Sequence[float]] = None,
    ) -> Tuple[float, List[dict]]:
        """Per-user threshold calibration (Sec. 3.6, "Calibration of the
        confidence threshold").

        ``variances``: Var(y_hat) for each calibration candidate.
        ``accepted``:  whether the simulated user accepted that candidate.
        ``confidences``: predicted acceptance probability per candidate. Needed
        only by the FAR guard; without it the search collapses to Sec. 3.6's
        one-dimensional sweep over tau.

        Searches (tau, floor) pairs and keeps those whose FAR respects the
        budget, preferring the one that admits the most candidates -- otherwise
        a floor of 1.0 would "meet" the budget by suppressing everything.
        Among equal-coverage pairs the smaller floor wins, so the printed rule
        (floor = 0) is never displaced without cause.

        Returns ``(tau, trace)``; ``trace`` records FAR at each point searched.
        ``self.budget_met`` says whether the budget was reachable at all.
        """
        v = np.asarray(variances, dtype=float)
        a = np.asarray(accepted, dtype=bool)
        n = max(v.size, 1)
        use_guard = self.guarded and confidences is not None
        c = np.asarray(confidences, dtype=float) if use_guard else np.ones_like(v)
        floors = self.floor_grid() if use_guard else np.array([0.0])

        trace: List[dict] = []
        feasible: List[tuple] = []  # (coverage, -floor, tau, floor)
        for tau in self.tau_grid():
            for floor in floors:
                passed = (v < tau) & (c >= floor)
                n_passed = int(passed.sum())
                far = float((passed & ~a).sum() / n_passed) if n_passed else 0.0
                coverage = n_passed / n
                row = {"tau": float(tau), "far": far, "n_passed": n_passed, "coverage": coverage}
                if use_guard:
                    row["floor"] = float(floor)
                trace.append(row)
                if n_passed > 0 and far <= self.cfg.far_budget:
                    feasible.append((coverage, -float(floor), float(tau), float(floor)))

        self.budget_met = bool(feasible)
        if not feasible:
            # Nothing meets the budget. Fall back to the tightest setting rather
            # than the loosest, and do not let the miss pass silently: a gate
            # reporting FAR far above budget is the signal that the scoring head
            # is confidently wrong, not that the threshold needs widening.
            self.tau = float(self.cfg.tau_grid_start)
            self.confidence_floor = float(floors[-1]) if use_guard else 0.0
            best_far = min((r["far"] for r in trace if r["n_passed"]), default=float("nan"))
            message = (
                f"FAR budget {self.cfg.far_budget:.0%} unreachable: best achievable FAR "
                f"over the search grid is {best_far:.1%}. Falling back to the tightest "
                f"setting (tau={self.tau}, floor={self.confidence_floor}). This usually "
                f"means the scoring head is confidently wrong rather than uncertain, "
                f"which no variance threshold can detect."
            )
            if getattr(self.cfg, "strict_far_budget", False):
                raise FARBudgetUnreachable(message)
            _logger.warning(message)
            return self.tau, trace

        if rule == "smallest":  # literal reading of Sec. 3.6, see D-01
            best = min(feasible, key=lambda t: (t[2], -t[0]))
        else:
            best = max(feasible)
        _, _, self.tau, floor = best
        self.confidence_floor = floor if use_guard else 0.0
        return self.tau, trace


def retrieval_mass_ok(node_scores, active_ids, ratio: float) -> bool:
    """Does the active sub-graph hold more attention mass than chance?

    Node scores are a softmax over ``|V|``, so ``K`` uniform nodes hold
    ``K/|V|`` of the mass. The sub-graph clears the floor when it holds at least
    ``ratio`` times that share, which is scale-free in both ``K`` and ``|V|``.

    This is a separate failure detector from the variance gate. Predictive
    variance cannot distinguish "the samples agree and are right" from "the
    samples agree and are wrong", so when retrieval is at chance the gate waves
    the turn through. Retrieval quality is observable directly, and this is
    where it is checked.

    ``ratio <= 0`` disables the check, which is the default. Accepts a torch
    tensor or any array exposing ``ndim``/``shape``.
    """
    if ratio <= 0.0 or node_scores is None or not len(active_ids):
        return True
    ndim = node_scores.dim() if hasattr(node_scores, "dim") else node_scores.ndim
    scores = node_scores[0] if ndim > 1 else node_scores
    n_nodes = int(scores.shape[0])
    if n_nodes == 0:
        return True
    mass = float(sum(float(scores[i]) for i in active_ids))
    chance = len(active_ids) / n_nodes
    return mass >= ratio * chance


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
