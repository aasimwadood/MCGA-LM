"""Bayesian uncertainty gate and threshold calibration (paper Sec. 3.6)."""

from __future__ import annotations

import numpy as np
import pytest

from mcga_lm.config import SafetyConfig
from mcga_lm.safety.gate import BayesianGate, clarification_bubbles


def test_decision_rule_is_strictly_below_tau() -> None:
    """Sec. 3.6: Var < tau -> present; Var >= tau -> suppress and clarify."""
    gate = BayesianGate(SafetyConfig(), tau=0.15)
    assert gate.decide(0.149).accept is True
    assert gate.decide(0.15).accept is False  # boundary belongs to suppression
    assert gate.decide(0.9).accept is False


def test_disabled_gate_presents_everything() -> None:
    """Ablation "\\ Bayesian Gate" (Sec. 4.4)."""
    gate = BayesianGate(SafetyConfig(enabled=False), tau=0.01)
    assert gate.decide(10.0).accept is True


def test_tau_grid_matches_section_3_6() -> None:
    """Sec. 3.6: tau in {0.02, 0.04, ..., 0.30}."""
    grid = BayesianGate(SafetyConfig()).tau_grid()
    assert grid[0] == pytest.approx(0.02)
    assert grid[-1] == pytest.approx(0.30)
    assert len(grid) == 15
    np.testing.assert_allclose(np.diff(grid), 0.02, atol=1e-9)


def test_far_is_monotone_in_tau() -> None:
    """The basis of discrepancy D-01: a larger tau can only admit more candidates."""
    rng = np.random.default_rng(0)
    variances = rng.uniform(0, 0.4, 400)
    accepted = variances < 0.10  # low-variance candidates are the good ones
    gate = BayesianGate(SafetyConfig())
    _, trace = gate.calibrate(variances, accepted)
    fars = [row["far"] for row in trace if row["n_passed"] > 0]
    assert all(b >= a - 1e-12 for a, b in zip(fars, fars[1:]))


def test_calibration_respects_the_far_budget_and_maximises_coverage() -> None:
    """Pick the largest tau whose FAR still respects the 5% budget (D-01)."""
    rng = np.random.default_rng(1)
    variances = rng.uniform(0, 0.4, 2000)
    # A separable population: low-variance candidates are accepted, with 1% label noise.
    accepted = (variances < 0.12) ^ (rng.random(2000) < 0.01)
    gate = BayesianGate(SafetyConfig())
    tau, trace = gate.calibrate(variances, accepted, rule="largest")
    chosen = next(r for r in trace if r["tau"] == pytest.approx(tau))
    assert chosen["far"] <= SafetyConfig().far_budget
    assert tau == pytest.approx(0.12, abs=0.02)  # the separation point
    bigger = [r for r in trace if r["tau"] > tau and r["n_passed"] > 0]
    assert all(r["far"] > SafetyConfig().far_budget for r in bigger)


def test_infeasible_calibration_falls_back_to_the_strictest_threshold() -> None:
    """If no grid point meets the budget, the gate takes the most conservative tau."""
    rng = np.random.default_rng(4)
    variances = rng.uniform(0, 0.4, 300)
    accepted = rng.random(300) < 0.5  # uncertainty carries no signal at all
    gate = BayesianGate(SafetyConfig())
    tau, trace = gate.calibrate(variances, accepted)
    assert tau == pytest.approx(SafetyConfig().tau_grid_start)
    assert all(r["far"] > SafetyConfig().far_budget for r in trace if r["n_passed"] > 0)


def test_literal_rule_of_section_3_6_picks_the_smallest_feasible_tau() -> None:
    """D-01: the paper's printed rule is implemented but is not the default."""
    rng = np.random.default_rng(2)
    variances = rng.uniform(0, 0.4, 400)
    accepted = rng.random(400) < np.clip(1.0 - variances * 3.0, 0, 1)
    smallest, _ = BayesianGate(SafetyConfig()).calibrate(variances, accepted, rule="smallest")
    largest, _ = BayesianGate(SafetyConfig()).calibrate(variances, accepted, rule="largest")
    assert smallest <= largest


def test_calibrated_tau_falls_inside_the_grid() -> None:
    rng = np.random.default_rng(3)
    v = rng.uniform(0, 0.5, 200)
    a = rng.random(200) < 0.5
    tau, _ = BayesianGate(SafetyConfig()).calibrate(v, a)
    assert 0.02 <= tau <= 0.30


def test_clarification_shows_three_simplified_bubbles() -> None:
    """Sec. 3.6: top-3 intent nodes as simplified "Intent Bubbles"."""
    prompt = clarification_bubbles(["pain_medication", "thirst", "bed_rail", "kettle"], [3, 7, 11, 19], k=3)
    assert prompt.bubbles == ["Pain", "Thirst", "Bed"]
    assert prompt.node_ids == [3, 7, 11]


# ---------------------------------------------------------------- D-16 ----- #
def test_calibration_grid_is_exactly_the_one_sec_3_6_prints() -> None:
    """Sec. 3.6: "compute FAR ... for tau in {0.02, 0.04, ..., 0.30}"."""
    grid = BayesianGate(SafetyConfig()).tau_grid()
    assert len(grid) == 15
    assert grid[0] == pytest.approx(0.02) and grid[-1] == pytest.approx(0.30)
    # Every point is an even hundredth.
    assert all(round(float(t) * 100) % 2 == 0 for t in grid)


@pytest.mark.parametrize("reported", [0.07, 0.21, 0.13, 0.15])
def test_reported_tau_values_are_not_on_the_grid(reported: float) -> None:
    """DEVIATION D-16: Sec. 3.6's own tau figures are unreachable by its grid.

    It reports default tau = 0.15 and a calibrated range of [0.07, 0.21] with
    mean 0.13. All are odd hundredths; the grid holds only even ones. The mean
    could fall between grid points, but the range endpoints cannot -- the min
    and max of a set of grid values are themselves grid values.
    """
    grid = BayesianGate(SafetyConfig()).tau_grid()
    assert not any(abs(float(t) - reported) < 1e-9 for t in grid)


def test_calibration_can_only_return_a_grid_point() -> None:
    """Which is why the range endpoints of Sec. 3.6 cannot be calibration output."""
    grid = set(round(float(t), 10) for t in BayesianGate(SafetyConfig()).tau_grid())
    rng = np.random.default_rng(0)
    for seed_variance in (0.01, 0.08, 0.25):
        variances = rng.normal(seed_variance, 0.02, 60).clip(0.001, 0.5).tolist()
        accepted = [v < seed_variance for v in variances]
        gate = BayesianGate(SafetyConfig())
        tau, _trace = gate.calibrate(variances, accepted)
        assert round(float(tau), 10) in grid
