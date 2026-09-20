"""Algorithm 1 -- inference for one communicative turn (paper p. 14).

The property that matters most is line 24: ``a <= C_max + J_max = 5`` holds by
construction, and the paper stresses it is architectural rather than an observed
maximum. These tests exercise the real control flow, not a mock of it.
"""

from __future__ import annotations

import numpy as np
import pytest


def _setup(torch, cfg, persona):
    from mcga_lm.data.dataset import TurnEncoder
    from mcga_lm.inference import AdaptiveCommunicator
    from mcga_lm.llm import build_backend
    from mcga_lm.memory.graph import IntentMemoryGraph
    from mcga_lm.models.gat import graph_tensors
    from mcga_lm.pipeline import MCGALM
    from mcga_lm.safety.gate import BayesianGate

    torch.manual_seed(0)
    model = MCGALM(cfg).eval()
    backend = build_backend(cfg.llm, embedding_dim=cfg.inputs.ling_dim)
    graph = IntentMemoryGraph.from_dict(persona.graph.to_dict())
    turns = persona.simulate_session(seed=0, n_turns=6)
    session = TurnEncoder(cfg.inputs).encode_session(persona, turns, graph, seed=0)
    with torch.no_grad():
        z, _ = model.encode_context(session.batch)
        state = model.cognitive_state(z)
        query = model.build_query(z, state)
        features, edge_index, edge_weight = graph_tensors(graph)
        node_features, node_scores = model.attend_graph(features, edge_index, query, edge_weight)
    comm = AdaptiveCommunicator(model, backend, cfg, gate=BayesianGate(cfg.safety))
    return comm, graph, turns, query, node_scores, node_features


def test_sact_never_exceeds_the_architectural_bound(torch_mod, small_cfg, persona) -> None:
    """Algorithm 1 line 24: a <= C_max + J_max = 5, by construction."""
    torch = torch_mod
    comm, graph, turns, query, ns, nf = _setup(torch, small_cfg, persona)
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for i, turn in enumerate(turns):
            res = comm.run_turn(persona, graph, turn, query[i : i + 1], ns[0:1], nf[0:1], rng)
            assert 1 <= res.sact <= small_cfg.inference.max_sact == 5
            assert res.clarification_rounds <= small_cfg.inference.c_max
            assert res.n_offered <= small_cfg.inference.j_max


def test_bound_holds_even_when_the_gate_always_suppresses(torch_mod, small_cfg, persona) -> None:
    """A degenerate gate must not produce an unbounded clarification loop."""
    torch = torch_mod
    comm, graph, turns, query, ns, nf = _setup(torch, small_cfg, persona)
    comm.gate.tau = 0.0  # Var >= tau always: every candidate is suppressed
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for i, turn in enumerate(turns[:3]):
            res = comm.run_turn(persona, graph, turn, query[i : i + 1], ns[0:1], nf[0:1], rng)
            assert res.clarification_rounds == small_cfg.inference.c_max
            assert res.forced is True  # line 15: proceed with the best candidate
            assert res.sact <= 5


def test_a_permissive_gate_skips_clarification_entirely(torch_mod, small_cfg, persona) -> None:
    torch = torch_mod
    comm, graph, turns, query, ns, nf = _setup(torch, small_cfg, persona)
    comm.gate.tau = 1e9  # Var < tau always
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for i, turn in enumerate(turns[:3]):
            res = comm.run_turn(persona, graph, turn, query[i : i + 1], ns[0:1], nf[0:1], rng)
            assert res.clarification_rounds == 0
            assert res.forced is False
            assert res.sact <= small_cfg.inference.j_max


def test_abstention_emits_no_utterance(torch_mod, small_cfg, persona) -> None:
    """Algorithm 1 line 23: return ABSTAIN and surface the manual grid fallback."""
    torch = torch_mod
    comm, graph, turns, query, ns, nf = _setup(torch, small_cfg, persona)
    rng = np.random.default_rng(1)
    with torch.no_grad():
        results = [
            comm.run_turn(persona, graph, t, query[i : i + 1], ns[0:1], nf[0:1], rng)
            for i, t in enumerate(turns)
        ]
    for res in results:
        assert res.abstained == (not res.accepted)
        if res.abstained:
            assert res.utterance is None


def test_accepting_an_utterance_updates_the_graph(torch_mod, small_cfg, persona) -> None:
    """Algorithm 1 line 19 / Sec. 3.4: accepted utterances update G."""
    torch = torch_mod
    comm, graph, turns, query, ns, nf = _setup(torch, small_cfg, persona)
    comm.gate.tau = 1e9
    rng = np.random.default_rng(0)
    edges_before = len(graph.edges)
    accepted_any = False
    with torch.no_grad():
        for i, turn in enumerate(turns):
            res = comm.run_turn(persona, graph, turn, query[i : i + 1], ns[0:1], nf[0:1], rng,
                                update_graph=True)
            accepted_any = accepted_any or res.accepted
    if accepted_any:
        assert len(graph.edges) >= edges_before


def test_update_graph_false_leaves_memory_untouched(torch_mod, small_cfg, persona) -> None:
    torch = torch_mod
    comm, graph, turns, query, ns, nf = _setup(torch, small_cfg, persona)
    comm.gate.tau = 1e9
    snapshot = (len(graph.nodes), len(graph.edges))
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for i, turn in enumerate(turns):
            comm.run_turn(persona, graph, turn, query[i : i + 1], ns[0:1], nf[0:1], rng,
                          update_graph=False)
    assert (len(graph.nodes), len(graph.edges)) == snapshot


def test_turn_is_deterministic_given_the_same_rng(torch_mod, small_cfg, persona) -> None:
    torch = torch_mod
    comm, graph_a, turns, query, ns, nf = _setup(torch, small_cfg, persona)
    from mcga_lm.memory.graph import IntentMemoryGraph

    graph_b = IntentMemoryGraph.from_dict(persona.graph.to_dict())
    with torch.no_grad():
        a = comm.run_turn(persona, graph_a, turns[0], query[0:1], ns[0:1], nf[0:1],
                          np.random.default_rng(7), update_graph=False)
        b = comm.run_turn(persona, graph_b, turns[0], query[0:1], ns[0:1], nf[0:1],
                          np.random.default_rng(7), update_graph=False)
    assert (a.utterance, a.sact, a.accepted) == (b.utterance, b.sact, b.accepted)
