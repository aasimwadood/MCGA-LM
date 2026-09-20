"""Config defaults must match the values the paper prints (Table 3, Sec. 3-4)."""

from __future__ import annotations

import pytest

from mcga_lm.config import Config, merge_overrides


def test_table3_hyperparameters(cfg: Config) -> None:
    # Table 3, verbatim.
    assert (cfg.perceiver.num_latents, cfg.perceiver.latent_dim) == (256, 512)
    assert cfg.perceiver.cross_heads == 8
    assert cfg.tft.window == 32
    assert cfg.tft.lstm_hidden == 128
    assert cfg.tft.attn_heads == 4
    assert cfg.graph.gat_heads == 4
    assert cfg.graph.node_dim == 300
    assert cfg.graph.top_k == 5
    assert (cfg.llm.lora_rank, cfg.llm.lora_alpha) == (16, 32)
    assert cfg.llm.temp_low_fatigue == 1.2
    assert cfg.llm.quantisation == "nf4"
    assert cfg.safety.mc_passes == 20


def test_other_paper_constants(cfg: Config) -> None:
    assert cfg.tft.state_dim == 64  # D_c, Sec. 3.3
    assert cfg.graph.half_life_days == 30.0  # Sec. 3.4
    assert cfg.llm.temp_high_fatigue == 0.5  # Sec. 3.5
    assert cfg.llm.top_p == 0.9  # Sec. 3.5
    assert cfg.llm.history_turns == 10  # Sec. 3.5 component 4
    assert cfg.inputs.ling_tokens == 10  # L, Sec. 3.2
    assert cfg.inputs.phys_rate_hz == 64  # Sec. 3.2
    assert cfg.safety.far_budget == 0.05  # Sec. 3.6 / 4.2
    assert cfg.loss.contrastive_temp == 0.07  # tau_c, Eq. 11
    assert (cfg.loss.lambda_contrastive, cfg.loss.lambda_recon, cfg.loss.lambda_intent) == (0.5, 0.3, 1.0)
    assert cfg.training.epochs == 100 and cfg.training.batch_size == 128  # Sec. 4.8
    assert cfg.training.lr == 1e-4


def test_algorithm1_bounds(cfg: Config) -> None:
    """Algorithm 1 lines 9 and 24: C_max = 2, J_max = 3, a <= 5."""
    assert cfg.inference.c_max == 2
    assert cfg.inference.j_max == 3
    assert cfg.inference.max_sact == 5


def test_persona_cohorts_sum(cfg: Config) -> None:
    """Sec. 4.1: 10 ALS + 6 cerebral palsy + 4 brainstem stroke = 20."""
    s = cfg.simulation
    assert s.n_als + s.n_cerebral_palsy + s.n_brainstem_stroke == s.n_personas == 20


def test_overrides_do_not_mutate_source(cfg: Config) -> None:
    updated = merge_overrides(cfg, ["training.epochs=7", "graph.enabled=false"])
    assert updated.training.epochs == 7 and updated.graph.enabled is False
    assert cfg.training.epochs == 100 and cfg.graph.enabled is True


def test_unknown_override_rejected(cfg: Config) -> None:
    with pytest.raises(ValueError):
        merge_overrides(cfg, ["graph.not_a_key=1"])


def test_yaml_roundtrip(tmp_path, cfg: Config) -> None:
    path = tmp_path / "cfg.yaml"
    cfg.save(path)
    assert Config.load(path).to_dict() == cfg.to_dict()
