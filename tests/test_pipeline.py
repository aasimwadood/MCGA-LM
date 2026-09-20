"""End-to-end wiring of Phases I-V (paper Sec. 3.1, Eq. 1)."""

from __future__ import annotations

import numpy as np
import pytest


def _session(cfg, persona, seed=0, n=6):
    from mcga_lm.data.dataset import TurnEncoder

    turns = persona.simulate_session(seed=seed, n_turns=n)
    return TurnEncoder(cfg.inputs).encode_session(persona, turns, persona.graph, seed=seed), turns


def test_forward_produces_every_stage_output(torch_mod, small_cfg, persona) -> None:
    from mcga_lm.models.gat import graph_tensors
    from mcga_lm.pipeline import MCGALM

    torch = torch_mod
    model = MCGALM(small_cfg).eval()
    session, _ = _session(small_cfg, persona)
    features, edge_index, edge_weight = graph_tensors(persona.graph)
    with torch.no_grad():
        out = model(session.batch, features, edge_index, edge_weight, with_reconstruction=True)
    n = session.n_turns
    assert out.z_ctx.shape == (n, small_cfg.perceiver.latent_dim)
    assert out.state.vector.shape == (n, small_cfg.tft.state_dim)
    assert out.query.shape == (n, model.query_dim)
    assert out.node_scores.shape == (n, len(persona.graph))
    assert out.reconstruction.shape == session.batch.phys.shape


def test_query_is_the_concatenation_of_context_and_state(torch_mod, small_cfg, persona) -> None:
    """Sec. 3.4: q_t = [z_ctx,t ; s_cog,t]."""
    from mcga_lm.pipeline import MCGALM

    torch = torch_mod
    model = MCGALM(small_cfg).eval()
    session, _ = _session(small_cfg, persona)
    with torch.no_grad():
        z, _ = model.encode_context(session.batch)
        state = model.cognitive_state(z)
        q = model.build_query(z, state)
    assert torch.allclose(q[:, : z.shape[1]], z)
    assert torch.allclose(q[:, z.shape[1] :], state.vector)


def test_gat_disabled_configuration_has_no_graph_stage(torch_mod, small_cfg, persona) -> None:
    """Ablation "\\ GAT" / the M-LLM baseline (Sec. 4.3-4.4)."""
    from mcga_lm.pipeline import MCGALM

    small_cfg.graph.enabled = False
    model = MCGALM(small_cfg).eval()
    assert model.gat is None
    session, _ = _session(small_cfg, persona)
    with torch_mod.no_grad():
        out = model(session.batch)
    assert out.node_scores is None


def test_intent_targets_point_at_real_graph_nodes(torch_mod, small_cfg, persona) -> None:
    """Eq. (13)'s y_i is a binary indicator over the persona's own graph."""
    session, turns = _session(small_cfg, persona)
    assert session.intent_targets.shape == (session.n_turns, len(persona.graph))
    for i, turn in enumerate(turns):
        positives = session.intent_targets[i].nonzero().flatten().tolist()
        names = {persona.graph.nodes[j].name.lower() for j in positives}
        assert names == {e.lower() for e in turn.intent.entities}


def test_training_step_reduces_the_intent_loss(torch_mod, small_cfg, persona) -> None:
    """A minimal optimisation sanity check on Eqs. (9) and (13)."""
    from mcga_lm.losses import total_loss
    from mcga_lm.models.gat import graph_tensors
    from mcga_lm.pipeline import MCGALM

    torch = torch_mod
    torch.manual_seed(0)
    model = MCGALM(small_cfg)
    session, _ = _session(small_cfg, persona, n=8)
    features, edge_index, edge_weight = graph_tensors(persona.graph)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-2)

    def step(update: bool) -> float:
        out = model(session.batch, features, edge_index, edge_weight, with_reconstruction=True)
        loss = total_loss(
            small_cfg.loss,
            z_ctx=out.z_ctx, z_utt=session.utterance_embeddings[:, : out.z_ctx.shape[1]],
            x_phys=session.batch.phys, x_recon=out.reconstruction,
            node_scores=out.node_scores, intent_targets=session.intent_targets,
        )
        if update:
            optimiser.zero_grad(set_to_none=True)
            loss.total.backward()
            optimiser.step()
        return float(loss.intent.detach())

    first = step(update=True)
    for _ in range(9):
        step(update=True)
    model.eval()
    with torch.no_grad():
        final = step(update=False)
    assert final < first


def test_scoring_context_width_matches_the_head(torch_mod, small_cfg, persona) -> None:
    from mcga_lm.models.gat import graph_tensors
    from mcga_lm.pipeline import MCGALM

    torch = torch_mod
    model = MCGALM(small_cfg).eval()
    session, _ = _session(small_cfg, persona)
    features, edge_index, edge_weight = graph_tensors(persona.graph)
    with torch.no_grad():
        out = model(session.batch, features, edge_index, edge_weight)
        sub = model.select_subgraph(out.node_scores, turn_index=0)
        context = model.scoring_context(out.query[0:1], out.node_features[0, sub.node_ids, :])
        tokens = torch.randn(1, 5, small_cfg.inputs.ling_dim)
        estimate = model.uncertainty(context, tokens)
    assert context.shape[1] == model.query_dim + model.graph_feature_dim
    assert estimate.variance.shape == (1,)
    assert len(sub.node_ids) == small_cfg.graph.top_k


def test_model_selector_accepts_a_shared_model_or_a_per_persona_mapping(torch_mod, small_cfg) -> None:
    """Sec. 3.7 describes a per-user on-device model; the runner supports both."""
    import pytest as _pytest

    from mcga_lm.eval.runner import _model_for
    from mcga_lm.pipeline import MCGALM

    shared = MCGALM(small_cfg)
    assert _model_for(shared, "P00") is shared
    assert _model_for(shared, "anything-at-all") is shared

    per_persona = {"P00": MCGALM(small_cfg), "P01": MCGALM(small_cfg)}
    assert _model_for(per_persona, "P01") is per_persona["P01"]
    with _pytest.raises(KeyError):
        _model_for(per_persona, "P99")


def test_per_persona_training_returns_one_model_and_one_tau_each(torch_mod, small_cfg, personas) -> None:
    """``train(per_persona=True)`` fits each persona in isolation (Sec. 3.7)."""
    from mcga_lm.llm import build_backend
    from mcga_lm.pipeline import MCGALM
    from mcga_lm.training.personalise import train

    small_cfg.simulation.turns_per_session = 4
    backend = build_backend(small_cfg.llm, embedding_dim=small_cfg.inputs.ling_dim)
    models, report = train(
        small_cfg, personas, backend, epochs=1, head_epochs=1, per_persona=True
    )
    ids = [p.spec.persona_id for p in personas]
    assert sorted(models) == sorted(ids)
    assert all(isinstance(m, MCGALM) for m in models.values())
    assert sorted(report.tau_by_persona) == sorted(ids)
    assert report.per_persona is True
    # The models are genuinely independent objects, not the same one aliased.
    assert len({id(m) for m in models.values()}) == len(models)
