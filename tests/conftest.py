"""Shared fixtures. Tests that need PyTorch skip cleanly when it is absent."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mcga_lm.config import Config  # noqa: E402


def _small(cfg: Config) -> Config:
    """Tiny but structurally identical configuration, for fast tests."""
    cfg.perceiver.num_latents = 8
    cfg.perceiver.latent_dim = 32
    cfg.perceiver.depth = 1
    cfg.perceiver.self_attn_per_block = 1
    cfg.perceiver.cross_heads = 4
    cfg.perceiver.self_heads = 4
    cfg.inputs.phys_window = 16
    cfg.inputs.beh_window = 4
    cfg.inputs.ling_dim = 32
    cfg.tft.window = 4
    cfg.tft.lstm_hidden = 16
    cfg.tft.state_dim = 8
    cfg.graph.node_dim = 16
    cfg.graph.gat_hidden = 8
    cfg.graph.target_nodes = 60
    cfg.safety.mc_passes = 4
    cfg.simulation.turns_per_session = 6
    cfg.simulation.n_personas = 2
    cfg.simulation.n_als = 1
    cfg.simulation.n_cerebral_palsy = 1
    cfg.simulation.n_brainstem_stroke = 0
    return cfg


@pytest.fixture
def cfg() -> Config:
    return Config()


@pytest.fixture
def small_cfg() -> Config:
    return _small(Config())


@pytest.fixture
def personas(small_cfg):
    from mcga_lm.data.personas import build_persona_suite

    return build_persona_suite(small_cfg.simulation, small_cfg.graph, seed=0)


@pytest.fixture
def persona(personas):
    return personas[0]


@pytest.fixture
def torch_mod():
    return pytest.importorskip("torch", reason="PyTorch is not installed in this environment")
