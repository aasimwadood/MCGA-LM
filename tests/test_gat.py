"""Graph Attention intent memory (paper Sec. 3.4, Eqs. 6-7)."""

from __future__ import annotations

import numpy as np
import pytest


def _graph(node_dim=16):
    from mcga_lm.memory.graph import IntentMemoryGraph

    g = IntentMemoryGraph(node_dim=node_dim)
    ids = {n: g.add_node(n, t) for n, t in
           (("Nurse_Smith", "Person"), ("kettle", "Object"), ("kitchen", "Object"),
            ("pain", "AbstractState"), ("physiotherapy", "Activity"))}
    g.add_edge(0, ids["Nurse_Smith"], "interacts_with", 0.9)
    g.add_edge(0, ids["pain"], "expressed_as", 0.8)
    g.add_edge(ids["kettle"], ids["kitchen"], "located_in", 0.7)
    g.add_edge(ids["physiotherapy"], ids["pain"], "causes", 0.6)
    return g


def test_segment_softmax_matches_a_dense_reference(torch_mod) -> None:
    from mcga_lm.models.gat import segment_softmax

    torch = torch_mod
    torch.manual_seed(0)
    n_segments, e = 4, 11
    index = torch.randint(0, n_segments, (e,))
    scores = torch.randn(2, e)
    out = segment_softmax(scores, index, n_segments)
    for b in range(2):
        for seg in range(n_segments):
            mask = index == seg
            if not bool(mask.any()):
                continue
            ref = torch.softmax(scores[b][mask], dim=0)
            assert torch.allclose(out[b][mask], ref, atol=1e-6)


def test_attention_coefficients_sum_to_one_per_target_node(torch_mod, small_cfg) -> None:
    """Eq. (7) is a softmax over the neighbourhood N_i."""
    from mcga_lm.models.gat import segment_softmax

    torch = torch_mod
    g = _graph()
    ei, _ = g.edge_index()
    dst = torch.from_numpy(ei[1])
    scores = torch.randn(3, dst.shape[0])
    alpha = segment_softmax(scores, dst, len(g))
    totals = torch.zeros(3, len(g))
    totals.index_add_(1, dst, alpha)
    assert torch.allclose(totals, torch.ones(3, len(g)), atol=1e-5)


def test_gat_output_shapes_and_normalised_node_scores(torch_mod, small_cfg) -> None:
    from mcga_lm.models.gat import GraphAttentionIntentMemory, graph_tensors

    torch = torch_mod
    g = _graph(node_dim=small_cfg.graph.node_dim)
    features, edge_index, edge_weight = graph_tensors(g)
    gat = GraphAttentionIntentMemory(query_dim=12, cfg=small_cfg.graph).eval()
    query = torch.randn(4, 12)
    h, scores = gat(features, edge_index, query, edge_weight)
    assert h.shape == (4, len(g), small_cfg.graph.gat_heads * small_cfg.graph.gat_hidden)
    assert scores.shape == (4, len(g))
    assert torch.allclose(scores.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert bool((scores >= 0).all())


def test_active_subgraph_is_the_top_K_by_attention(torch_mod, small_cfg) -> None:
    """Sec. 3.4: "the top-K nodes with the highest aggregated attention scores"."""
    from mcga_lm.models.gat import GraphAttentionIntentMemory

    torch = torch_mod
    gat = GraphAttentionIntentMemory(query_dim=8, cfg=small_cfg.graph)
    scores = torch.tensor([[0.05, 0.4, 0.01, 0.3, 0.2, 0.04]])
    sub = gat.select_active_subgraph(scores, top_k=3)
    assert sub.node_ids == [1, 3, 4]
    assert sub.top_scores() == pytest.approx([0.4, 0.3, 0.2])


def test_top_k_defaults_to_the_paper_value(torch_mod, cfg) -> None:
    from mcga_lm.models.gat import GraphAttentionIntentMemory

    torch = torch_mod
    gat = GraphAttentionIntentMemory(query_dim=8, cfg=cfg.graph)
    scores = torch.rand(1, 50)
    assert len(gat.select_active_subgraph(scores).node_ids) == 5  # K_top = 5, Table 3


def test_attention_is_query_dependent(torch_mod, small_cfg) -> None:
    """Sec. 3.4: "the query-dependent attention ensures that only nodes relevant
    to the current conversation and user state receive high scores"."""
    from mcga_lm.models.gat import GraphAttentionIntentMemory, graph_tensors

    torch = torch_mod
    torch.manual_seed(0)
    g = _graph(node_dim=small_cfg.graph.node_dim)
    features, edge_index, edge_weight = graph_tensors(g)
    gat = GraphAttentionIntentMemory(query_dim=12, cfg=small_cfg.graph).eval()
    q = torch.randn(2, 12)
    q[1] = q[0] * -3.0
    _, scores = gat(features, edge_index, q, edge_weight)
    assert not torch.allclose(scores[0], scores[1], atol=1e-6)


def test_edge_weight_biases_attention_towards_strong_associations(torch_mod, small_cfg) -> None:
    """Sec. 3.4: weights reflect association strength and recency (A-09)."""
    from mcga_lm.models.gat import GraphAttentionIntentMemory, graph_tensors

    torch = torch_mod
    torch.manual_seed(0)
    g = _graph(node_dim=small_cfg.graph.node_dim)
    features, edge_index, edge_weight = graph_tensors(g)
    gat = GraphAttentionIntentMemory(query_dim=12, cfg=small_cfg.graph).eval()
    q = torch.randn(1, 12)
    # Edge weights enter Eq. (7)'s logits, so they move the message-passing
    # output h' of Eq. (6). (Under the default query-based node scoring the
    # selection scores are computed from q and h directly and are unaffected.)
    with_weights, _ = gat(features, edge_index, q, edge_weight)
    without, _ = gat(features, edge_index, q, None)
    assert not torch.allclose(with_weights, without, atol=1e-6)


def test_gradients_reach_the_attention_vectors(torch_mod, small_cfg) -> None:
    from mcga_lm.models.gat import GraphAttentionIntentMemory, graph_tensors

    g = _graph(node_dim=small_cfg.graph.node_dim)
    features, edge_index, edge_weight = graph_tensors(g)
    gat = GraphAttentionIntentMemory(query_dim=12, cfg=small_cfg.graph)
    h_prime, scores = gat(features, edge_index, torch_mod.randn(2, 12), edge_weight)
    (h_prime.sum() + scores.sum()).backward()
    for name in ("attn_q", "attn_i", "attn_j"):
        grad = getattr(gat, name).grad
        assert grad is not None and torch_mod.isfinite(grad).all()


def test_printed_equation_7_is_additively_separable(torch_mod, small_cfg) -> None:
    """DISCREPANCY D-06.

    ``a^T [W_q q || W_h h_i || W_h h_j]`` expands to three independent dot
    products, so the query contributes a constant within each softmax segment.
    Without the LeakyReLU it would cancel exactly; with it, attention barely
    moves when the query changes. This test documents the finding.
    """
    from mcga_lm.models.gat import GraphAttentionIntentMemory, graph_tensors

    torch = torch_mod
    torch.manual_seed(0)
    small_cfg.graph.attention_form = "paper"
    small_cfg.graph.node_scoring = "incoming"  # look at the attention-derived scores
    small_cfg.graph.negative_slope = 1.0  # LeakyReLU becomes the identity
    g = _graph(node_dim=small_cfg.graph.node_dim)
    features, edge_index, edge_weight = graph_tensors(g)
    gat = GraphAttentionIntentMemory(query_dim=12, cfg=small_cfg.graph).eval()
    q = torch.randn(2, 12)
    q[1] = q[0] * 10.0 + 3.0  # a wildly different query
    _, scores = gat(features, edge_index, q, edge_weight)
    assert torch.allclose(scores[0], scores[1], atol=1e-5), (
        "with a linear activation the printed Eq. (7) is exactly query-independent"
    )


def test_query_gated_form_restores_query_dependence(torch_mod, small_cfg) -> None:
    """The minimal fix of D-06: a multiplicative query-neighbour term."""
    from mcga_lm.models.gat import GraphAttentionIntentMemory, graph_tensors

    torch = torch_mod
    torch.manual_seed(0)
    small_cfg.graph.attention_form = "query_gated"
    small_cfg.graph.node_scoring = "incoming"
    small_cfg.graph.negative_slope = 1.0
    g = _graph(node_dim=small_cfg.graph.node_dim)
    features, edge_index, edge_weight = graph_tensors(g)
    gat = GraphAttentionIntentMemory(query_dim=12, cfg=small_cfg.graph).eval()
    q = torch.randn(2, 12)
    q[1] = q[0] * 10.0 + 3.0
    _, scores = gat(features, edge_index, q, edge_weight)
    assert not torch.allclose(scores[0], scores[1], atol=1e-3)


def test_unknown_attention_form_is_rejected(torch_mod, small_cfg) -> None:
    from mcga_lm.models.gat import GraphAttentionIntentMemory

    small_cfg.graph.attention_form = "nonsense"
    with pytest.raises(ValueError):
        GraphAttentionIntentMemory(query_dim=8, cfg=small_cfg.graph)


def test_query_node_scoring_is_degree_free_while_incoming_is_not(torch_mod, small_cfg) -> None:
    """DISCREPANCY D-07.

    Summing received attention ranks nodes largely by how many neighbours they
    have; scoring by how strongly the query reaches for a node does not. The
    star graph below makes the difference visible: its hub has every edge.
    """
    from mcga_lm.memory.graph import IntentMemoryGraph
    from mcga_lm.models.gat import GraphAttentionIntentMemory, graph_tensors

    torch = torch_mod
    torch.manual_seed(0)
    g = IntentMemoryGraph(node_dim=small_cfg.graph.node_dim)
    hub = g.add_node("hub", "Object")
    for i in range(8):
        leaf = g.add_node(f"leaf_{i}", "Object")
        g.add_edge(hub, leaf, "associated_with", weight=0.8)
    features, edge_index, edge_weight = graph_tensors(g)
    q = torch.randn(1, 12)

    small_cfg.graph.node_scoring = "incoming"
    _, incoming = GraphAttentionIntentMemory(12, small_cfg.graph).eval()(
        features, edge_index, q, edge_weight
    )
    small_cfg.graph.node_scoring = "query"
    _, by_query = GraphAttentionIntentMemory(12, small_cfg.graph).eval()(
        features, edge_index, q, edge_weight
    )
    assert int(incoming.argmax()) == hub, "incoming attention concentrates on the hub"
    # Query scoring has no structural reason to favour the hub.
    assert float(by_query.max()) < float(incoming.max())
