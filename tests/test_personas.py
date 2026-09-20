"""Synthetic persona suite (paper Sec. 4.1) and the fatigue model (Sec. 4.5)."""

from __future__ import annotations

import numpy as np
import pytest

from mcga_lm.config import Config
from mcga_lm.data.personas import build_persona_suite
from mcga_lm.data.physiology import FatigueModel
from mcga_lm.data.taxonomy import PRAGMATIC_FUNCTIONS


def test_suite_composition_matches_section_4_1() -> None:
    cfg = Config()
    personas = build_persona_suite(cfg.simulation, cfg.graph, seed=0)
    assert len(personas) == 20
    cohorts = [p.spec.cohort for p in personas]
    assert cohorts.count("ALS") == 10
    assert cohorts.count("CP") == 6
    assert cohorts.count("STROKE") == 4


def test_graphs_are_dense_and_within_the_steady_state_band() -> None:
    """Sec. 4.1: "~400 nodes"; Sec. 3.4: steady state 200-500 nodes."""
    cfg = Config()
    personas = build_persona_suite(cfg.simulation, cfg.graph, seed=0)
    sizes = [len(p.graph) for p in personas]
    assert all(cfg.graph.min_nodes <= n <= cfg.graph.max_nodes for n in sizes)
    assert 300 <= float(np.mean(sizes)) <= 500


def test_personas_are_distinct() -> None:
    cfg = Config()
    personas = build_persona_suite(cfg.simulation, cfg.graph, seed=0)
    people = [frozenset(n for n, _ in p.spec.people) for p in personas]
    assert len(set(people)) == len(people)


def test_taxonomy_has_24_functions() -> None:
    assert len(PRAGMATIC_FUNCTIONS) == 24
    assert len(set(PRAGMATIC_FUNCTIONS)) == 24


def test_session_shape_and_grounded_intents(persona) -> None:
    turns = persona.simulate_session(seed=0, n_turns=12)
    assert len(turns) == 12
    for t in turns:
        assert t.intent.function in PRAGMATIC_FUNCTIONS
        assert t.intent.entities and t.intent.reference
        # Every ground-truth entity must exist in the persona's own graph.
        assert all(persona.graph.has_node(e) for e in t.intent.entities)


def test_sessions_are_reproducible(persona) -> None:
    a = persona.simulate_session(seed=3, n_turns=8)
    b = persona.simulate_session(seed=3, n_turns=8)
    assert [t.intent.reference for t in a] == [t.intent.reference for t in b]
    c = persona.simulate_session(seed=4, n_turns=8)
    assert [t.intent.reference for t in a] != [t.intent.reference for t in c]


def test_fatigue_rises_over_the_session(persona) -> None:
    turns = persona.simulate_session(seed=0, n_turns=40)
    early = np.mean([t.context.fatigue for t in turns[:8]])
    late = np.mean([t.context.fatigue for t in turns[-8:]])
    assert late > early


def test_fatigue_model_half_life_semantics() -> None:
    """Sec. 4.5: exponential rise; at t = half-life the rise is half-complete."""
    m = FatigueModel(half_life_min=30.0, peak=0.8, baseline=0.0, noise=0.0)
    assert float(m.value(0.0)) == pytest.approx(0.0, abs=1e-9)
    assert float(m.value(30.0)) == pytest.approx(0.4, abs=1e-9)
    assert float(m.value(600.0)) == pytest.approx(0.8, abs=1e-3)
    assert float(m.value(60.0)) > float(m.value(30.0))


def test_adaptation_slows_the_rise(persona) -> None:
    """Fig. 5: TFT adaptation slows the fatigue rise after activation."""
    m = FatigueModel(half_life_min=30.0, peak=0.8, noise=0.0)
    t = np.linspace(0, 60, 61)
    base, adapted = m.value(t), m.adapted_value(t)
    assert np.all(adapted <= base + 1e-9)
    assert adapted[-1] < base[-1]
    np.testing.assert_allclose(adapted[:10], base[:10], atol=1e-9)  # identical before onset


def test_intake_graph_is_a_subset_containing_all_people(persona) -> None:
    """Sec. 3.7: the graph is initialised from a structured intake interview."""
    intake = persona.intake_graph(fraction=0.2, seed=0)
    assert len(intake) < len(persona.graph)
    for name, _ in persona.spec.people:
        assert intake.has_node(name)


def test_acceptance_model(persona) -> None:
    from mcga_lm.data.personas import GroundTruthIntent

    rng = np.random.default_rng(0)
    truth = GroundTruthIntent(function="express_pain", entities=("pain",), reference="x")
    assert persona.accepts("express_pain", ("pain",), truth, rng) is True
    assert persona.accepts("express_pain", ("kettle",), truth, rng) is False
    # entity match with the wrong function is accepted only sometimes
    hits = [persona.accepts("request_object", ("pain",), truth, rng) for _ in range(400)]
    assert 0.15 < float(np.mean(hits)) < 0.45
