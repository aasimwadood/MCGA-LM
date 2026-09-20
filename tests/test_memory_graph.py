"""Dynamic User Intent Graph (paper Sec. 3.4)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mcga_lm.memory.graph import (
    NODE_TYPES,
    RELATIONS,
    USER_NODE,
    IntentMemoryGraph,
    SemanticFrame,
    embed_lexeme,
)


def test_paper_node_types_and_relations() -> None:
    """Sec. 3.4 lists exactly four node categories and six typed relations."""
    assert set(NODE_TYPES) == {"Person", "Object", "Activity", "AbstractState"}
    assert set(RELATIONS) == {
        "interacts_with", "located_in", "causes", "associated_with", "precedes", "expressed_as",
    }


def test_embeddings_are_300d_and_deterministic() -> None:
    a = embed_lexeme("nurse_smith", 300)
    b = embed_lexeme("Nurse_Smith", 300)  # case-insensitive
    assert a.shape == (300,)
    np.testing.assert_allclose(a, b)
    assert not np.allclose(a, embed_lexeme("kettle", 300))


def test_user_root_exists() -> None:
    g = IntentMemoryGraph(node_dim=8)
    assert len(g) == 1 and g.nodes[0].name == USER_NODE


def test_half_life_decay_is_exactly_30_days() -> None:
    """Sec. 3.4: weights decay with a 30-day half-life unless reinforced."""
    g = IntentMemoryGraph(node_dim=8, half_life_days=30.0)
    a = g.add_node("pain", "AbstractState")
    g.add_edge(0, a, "expressed_as", weight=0.8)
    g.decay_to(30.0)
    assert g.edges[0].weight == pytest.approx(0.4, abs=1e-9)
    g.decay_to(60.0)
    assert g.edges[0].weight == pytest.approx(0.2, abs=1e-9)


def test_reinforcement_saturates_in_unit_interval() -> None:
    g = IntentMemoryGraph(node_dim=8)
    a = g.add_node("kettle", "Object")
    g.add_edge(0, a, "associated_with", weight=0.5)
    for _ in range(50):
        g.reinforce(0, amount=0.3)
    assert 0.0 <= g.edges[0].weight <= 1.0
    assert g.edges[0].weight == pytest.approx(1.0, abs=1e-6)


def test_pruning_removes_weak_edges_and_isolated_nodes() -> None:
    g = IntentMemoryGraph(node_dim=8, prune_threshold=0.05, half_life_days=30.0)
    keep = g.add_node("kettle", "Object")
    drop = g.add_node("ephemeral", "Object")
    g.add_edge(0, keep, "associated_with", weight=0.9)
    g.add_edge(0, drop, "associated_with", weight=0.06)
    g.decay_to(120.0)  # four half-lives: 0.06 -> 0.00375, 0.9 -> 0.05625
    removed = g.prune()
    assert removed == 1
    assert g.has_node("kettle") and not g.has_node("ephemeral")
    assert g.nodes[0].name == USER_NODE  # the root is never dropped


def test_frame_update_adds_nodes_edges_and_partner() -> None:
    g = IntentMemoryGraph(node_dim=8)
    g.update_from_frame(
        SemanticFrame(predicate="expressed_as", obj="pain", obj_type="AbstractState",
                      partner="Nurse_Smith", intent_nodes=("pain",))
    )
    assert g.has_node("pain") and g.has_node("Nurse_Smith")
    assert g.neighbour_weight(USER_NODE, "Nurse_Smith") > 0


def test_edge_index_is_symmetric_with_self_loops() -> None:
    g = IntentMemoryGraph(node_dim=8)
    a = g.add_node("kettle", "Object")
    g.add_edge(0, a, "associated_with", weight=0.5)
    ei, ew = g.edge_index()
    assert ei.shape[0] == 2 and ei.shape[1] == ew.shape[0]
    # one self-loop per node plus both directions of the single edge
    assert ei.shape[1] == len(g) + 2
    pairs = set(zip(ei[0].tolist(), ei[1].tolist()))
    assert (0, a) in pairs and (a, 0) in pairs
    assert all((i, i) in pairs for i in range(len(g)))


def test_serialisation_roundtrip() -> None:
    g = IntentMemoryGraph(node_dim=16)
    a = g.add_node("kettle", "Object")
    b = g.add_node("kitchen", "Object")
    g.add_edge(a, b, "located_in", weight=0.7)
    restored = IntentMemoryGraph.from_dict(g.to_dict())
    assert restored.node_names() == g.node_names()
    assert [e.weight for e in restored.edges] == [e.weight for e in g.edges]


def test_storage_estimate_matches_section_4_10() -> None:
    """Sec. 4.10: |V| . d . 4 bytes = 500 x 300 x 4 ~= 0.6 MB for node embeddings."""
    g = IntentMemoryGraph(node_dim=300)
    for i in range(499):
        g.add_node(f"node_{i}", "Object")
    assert len(g) == 500
    node_bytes = 500 * 300 * 4
    assert node_bytes == pytest.approx(0.6e6, rel=0.01)
    assert g.storage_bytes() / 1e6 < 1.0  # "well under 1 MB per user"


def test_invalid_type_and_relation_rejected() -> None:
    g = IntentMemoryGraph(node_dim=8)
    with pytest.raises(ValueError):
        g.add_node("x", "NotAType")
    a = g.add_node("x", "Object")
    with pytest.raises(ValueError):
        g.add_edge(0, a, "not_a_relation")
