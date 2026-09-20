"""Seed control and reproducibility (paper Sec. 3.8: five seeded replicates)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from mcga_lm.seed import rng, set_seed, temporary_seed


def test_set_seed_makes_numpy_and_random_reproducible() -> None:
    set_seed(1234)
    a = (np.random.rand(5).tolist(), [random.random() for _ in range(5)])
    set_seed(1234)
    b = (np.random.rand(5).tolist(), [random.random() for _ in range(5)])
    assert a == b


def test_independent_generators_do_not_share_state() -> None:
    a, b = rng(7), rng(7)
    np.testing.assert_allclose(a.random(4), b.random(4))
    assert not np.allclose(rng(7).random(4), rng(8).random(4))


def test_temporary_seed_restores_global_state() -> None:
    set_seed(0)
    before = np.random.rand(3).tolist()
    set_seed(0)
    np.random.rand(3)
    state = np.random.get_state()[1][:5].tolist()
    with temporary_seed(999):
        np.random.rand(10)
    assert np.random.get_state()[1][:5].tolist() == state
    set_seed(0)
    assert np.random.rand(3).tolist() == before


def test_torch_is_seeded_when_available(torch_mod) -> None:
    set_seed(42)
    a = torch_mod.randn(4)
    set_seed(42)
    assert torch_mod.equal(a, torch_mod.randn(4))


def test_persona_suite_is_reproducible_across_processes() -> None:
    from mcga_lm.config import Config
    from mcga_lm.data.personas import build_persona_suite

    cfg = Config()
    a = build_persona_suite(cfg.simulation, cfg.graph, seed=5)
    b = build_persona_suite(cfg.simulation, cfg.graph, seed=5)
    assert [p.spec.people for p in a] == [p.spec.people for p in b]
    assert [p.graph.node_names() for p in a] == [p.graph.node_names() for p in b]
    c = build_persona_suite(cfg.simulation, cfg.graph, seed=6)
    assert [p.spec.people for p in a] != [p.spec.people for p in c]
