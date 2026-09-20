"""Bayesian uncertainty gate and threshold calibration (paper Sec. 3.6)."""

from __future__ import annotations

import numpy as np
import pytest

from mcga_lm.config import SafetyConfig
from mcga_lm.safety.gate import BayesianGate, FARBudgetUnreachable, clarification_bubbles


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


# ------------------------------------------------- FAR guard (D-17) -------- #
def _confidently_wrong(n: int = 400, seed: int = 0):
    """Variance says "certain", the user says "no". The case Sec. 3.6 cannot see."""
    rng = np.random.default_rng(seed)
    variances = rng.uniform(0.0, 0.005, n).tolist()   # below every grid point
    confidences = rng.uniform(0.80, 0.99, n).tolist()  # confident about all of it
    accepted = [bool(rng.random() < 0.001) for _ in range(n)]
    return variances, confidences, accepted


def _separable(n: int = 600, seed: int = 1):
    """Variance is uninformative but confidence tracks correctness."""
    rng = np.random.default_rng(seed)
    good = rng.random(n) < 0.35
    variances = rng.uniform(0.0, 0.02, n)
    confidences = np.where(good, rng.uniform(0.6, 0.95, n), rng.uniform(0.05, 0.5, n))
    return variances.tolist(), confidences.tolist(), good.tolist()


def test_printed_rule_ignores_confidence_entirely() -> None:
    """Sec. 3.6 gates on Var(y_hat) < tau alone; this pins that reading."""
    cfg = SafetyConfig()
    cfg.decision_rule = "variance_only"
    gate = BayesianGate(cfg, tau=0.10)
    assert gate.decide(variance=0.01, confidence=0.99).accept
    assert gate.decide(variance=0.01, confidence=0.00).accept, "confidence must not matter"


def test_variance_alone_cannot_bound_far_when_the_head_is_confidently_wrong() -> None:
    """The defect the guard exists to fix.

    With every variance below the smallest grid point, no tau suppresses
    anything, so FAR stays pinned near 100% across the entire grid and the
    budget is unreachable.
    """
    cfg = SafetyConfig()
    cfg.decision_rule = "variance_only"
    variances, confidences, accepted = _confidently_wrong()
    gate = BayesianGate(cfg)
    _tau, trace = gate.calibrate(variances, accepted, confidences=confidences)
    assert gate.budget_met is False
    assert all(r["n_passed"] == len(variances) for r in trace), "every tau admits everything"
    assert min(r["far"] for r in trace) > 0.9


def test_guard_reaches_the_far_budget_where_the_printed_rule_cannot() -> None:
    """Same data, two rules: the floor is what makes the budget attainable."""
    variances, confidences, accepted = _separable()

    literal = SafetyConfig()
    literal.decision_rule = "variance_only"
    g1 = BayesianGate(literal)
    g1.calibrate(variances, accepted, confidences=confidences)
    far1 = _empirical_far(g1, variances, confidences, accepted)

    guarded = SafetyConfig()  # 'guarded' is the default
    g2 = BayesianGate(guarded)
    g2.calibrate(variances, accepted, confidences=confidences)
    far2 = _empirical_far(g2, variances, confidences, accepted)

    assert g1.budget_met is False and far1 > 0.05
    assert g2.budget_met is True
    assert far2 <= guarded.far_budget
    assert g2.confidence_floor > 0.0, "the guard must actually use a floor here"


def _empirical_far(gate: BayesianGate, variances, confidences, accepted) -> float:
    passed = [(a) for v, c, a in zip(variances, confidences, accepted) if gate.decide(v, c).accept]
    return (sum(1 for a in passed if not a) / len(passed)) if passed else 0.0


def test_guard_does_not_trivially_suppress_everything() -> None:
    """A floor of 1.0 would "meet" the budget by admitting nothing.

    Calibration maximises coverage among feasible settings precisely so that
    degenerate solution is never selected when a useful one exists.
    """
    variances, confidences, accepted = _separable()
    gate = BayesianGate(SafetyConfig())
    gate.calibrate(variances, accepted, confidences=confidences)
    n_passed = sum(1 for v, c in zip(variances, confidences) if gate.decide(v, c).accept)
    assert n_passed > 0.1 * len(variances), "a useful gate must still admit candidates"


def test_unreachable_budget_falls_back_to_the_tightest_setting() -> None:
    """Failing safe means tightening, not widening."""
    variances, confidences, accepted = _confidently_wrong()
    cfg = SafetyConfig()
    gate = BayesianGate(cfg)
    tau, _ = gate.calibrate(variances, accepted, confidences=confidences)
    assert gate.budget_met is False
    assert tau == pytest.approx(cfg.tau_grid_start), "tightest tau, not the default"
    assert gate.confidence_floor == pytest.approx(cfg.floor_grid_stop), "highest floor"


def test_strict_mode_raises_when_the_budget_is_unreachable() -> None:
    variances, confidences, accepted = _confidently_wrong()
    cfg = SafetyConfig()
    cfg.strict_far_budget = True
    with pytest.raises(FARBudgetUnreachable):
        BayesianGate(cfg).calibrate(variances, accepted, confidences=confidences)


def test_budget_met_is_none_before_calibration() -> None:
    """An uncalibrated gate must not look like one that passed its budget."""
    assert BayesianGate(SafetyConfig()).budget_met is None


def test_guard_is_inert_without_confidences() -> None:
    """Calibrating without confidences collapses to Sec. 3.6's 1-D sweep."""
    rng = np.random.default_rng(3)
    variances = rng.uniform(0.0, 0.3, 200).tolist()
    accepted = [v < 0.1 for v in variances]
    gate = BayesianGate(SafetyConfig())
    gate.calibrate(variances, accepted)
    assert gate.confidence_floor == 0.0


def test_disabled_gate_still_presents_everything() -> None:
    """The "\\ Bayesian Gate" ablation must not be resurrected by the guard."""
    cfg = SafetyConfig()
    cfg.enabled = False
    gate = BayesianGate(cfg, tau=0.01, confidence_floor=0.9)
    assert gate.decide(variance=99.0, confidence=0.0).accept
