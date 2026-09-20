"""Retrieval floor: abstain when the active sub-graph is no better than chance.

The Bayesian gate keys on predictive variance, which cannot distinguish "the
samples agree and are right" from "the samples agree and are wrong". When
retrieval is at chance the sub-graph is arbitrary, the scoring head is
confidently wrong, and the gate waves the turn through. This guard is a second
trigger for the same Clarification Mode, keyed on retrieval quality instead.

``min_retrieval_mass_ratio`` defaults to 0.0 (disabled), so these tests pass the
ratio explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest

from mcga_lm.config import Config
from mcga_lm.safety.gate import retrieval_mass_ok

ACTIVE = [0, 1, 2, 3, 4]


def _scores(n_nodes: int, active, mass: float) -> np.ndarray:
    """A score vector placing ``mass`` on ``active`` and spreading the rest."""
    rest = (1.0 - mass) / (n_nodes - len(active))
    s = np.full((1, n_nodes), rest, dtype=float)
    for i in active:
        s[0, i] = mass / len(active)
    return s


def test_floor_is_disabled_by_default() -> None:
    """The default must not change any existing behaviour."""
    assert Config().inference.min_retrieval_mass_ratio == 0.0
    at_chance = _scores(400, ACTIVE, 5 / 400)
    assert retrieval_mass_ok(at_chance, ACTIVE, ratio=0.0)


def test_chance_level_retrieval_fails_the_floor() -> None:
    """5 of 400 nodes holding 5/400 of the mass is exactly uniform."""
    at_chance = _scores(400, ACTIVE, 5 / 400)
    assert not retrieval_mass_ok(at_chance, ACTIVE, ratio=2.0)


def test_concentrated_retrieval_clears_the_floor() -> None:
    concentrated = _scores(400, ACTIVE, 0.60)  # ~48x chance
    assert retrieval_mass_ok(concentrated, ACTIVE, ratio=2.0)


def test_floor_is_scale_free_in_graph_size() -> None:
    """The same multiple of chance must decide the same way at any |V|."""
    for n in (100, 400, 1000):
        under = _scores(n, ACTIVE, 2.5 * len(ACTIVE) / n)
        over = _scores(n, ACTIVE, 3.5 * len(ACTIVE) / n)
        assert not retrieval_mass_ok(under, ACTIVE, ratio=3.0), f"|V|={n}"
        assert retrieval_mass_ok(over, ACTIVE, ratio=3.0), f"|V|={n}"


def test_floor_is_scale_free_in_subgraph_size() -> None:
    """And the same at any K, since chance is K/|V|."""
    for k in (1, 3, 5, 10):
        active = list(range(k))
        under = _scores(400, active, 1.5 * k / 400)
        over = _scores(400, active, 4.0 * k / 400)
        assert not retrieval_mass_ok(under, active, ratio=2.0), f"K={k}"
        assert retrieval_mass_ok(over, active, ratio=2.0), f"K={k}"


def test_a_one_dimensional_score_vector_is_accepted() -> None:
    """Callers may pass a batched (1, |V|) array or a bare (|V|,) one."""
    flat = _scores(400, ACTIVE, 0.60)[0]
    assert retrieval_mass_ok(flat, ACTIVE, ratio=2.0)


@pytest.mark.parametrize("ratio", [2.0, 3.0, 10.0])
def test_absent_retrieval_does_not_block_the_turn(ratio: float) -> None:
    """No scores at all is the "\\GAT" ablation, not a safety event."""
    assert retrieval_mass_ok(None, ACTIVE, ratio=ratio)
    assert retrieval_mass_ok(_scores(10, [0], 0.5), [], ratio=ratio)


def test_abstain_reason_defaults_to_empty(torch_mod) -> None:
    """Accepted turns carry no abstention reason."""
    from mcga_lm.inference import TurnResult

    result = TurnResult(
        utterance="yes please", sact=1, accepted=True, abstained=False,
        clarification_rounds=0, variance=0.0, confidence=0.9,
        gate_accepted=True, forced=False,
    )
    assert result.abstain_reason == ""
