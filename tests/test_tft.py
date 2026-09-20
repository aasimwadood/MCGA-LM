"""Temporal Fusion Transformer (paper Sec. 3.3, Eq. 5)."""

from __future__ import annotations

import pytest


def test_cognitive_state_vector_has_dimension_D_c(torch_mod, small_cfg) -> None:
    """Sec. 3.3: s_cog,t in R^{D_c}."""
    from mcga_lm.models.tft import TemporalFusionTransformer

    tft = TemporalFusionTransformer(small_cfg.perceiver.latent_dim, small_cfg.tft).eval()
    seq = torch_mod.randn(5, small_cfg.tft.window, small_cfg.perceiver.latent_dim)
    state = tft(seq)
    assert state.vector.shape == (5, small_cfg.tft.state_dim)
    assert state.fatigue.shape == (5,)


def test_state_scalars_are_bounded_in_unit_interval(torch_mod, small_cfg) -> None:
    from mcga_lm.models.tft import TemporalFusionTransformer

    tft = TemporalFusionTransformer(small_cfg.perceiver.latent_dim, small_cfg.tft).eval()
    state = tft(torch_mod.randn(8, small_cfg.tft.window, small_cfg.perceiver.latent_dim))
    for tensor in (state.fatigue, state.cognitive_load, state.arousal):
        assert bool(((tensor >= 0) & (tensor <= 1)).all())


def test_variable_selection_weights_are_a_simplex(torch_mod, small_cfg) -> None:
    """Eq. (5): v_t^{(d)} are "softmax-normalised weights"."""
    from mcga_lm.models.tft import VariableSelectionNetwork

    vsn = VariableSelectionNetwork(16, 8).eval()
    x = torch_mod.randn(3, 4, 16)
    selected, weights = vsn(x)
    assert weights.shape == x.shape
    assert torch_mod.allclose(weights.sum(dim=-1), torch_mod.ones(3, 4), atol=1e-5)
    assert bool((weights >= 0).all())
    assert selected.shape == x.shape  # dimensionality is preserved (see A-03 / D note)


def test_variable_selection_prunes_uninformative_dimensions(torch_mod, small_cfg) -> None:
    from mcga_lm.models.tft import VariableSelectionNetwork

    vsn = VariableSelectionNetwork(16, 8).eval()
    _, weights = vsn(torch_mod.randn(2, 3, 16))
    # A learned selection is not uniform.
    assert float(weights.std()) > 0


def test_disabled_tft_returns_the_low_fatigue_state(torch_mod, small_cfg) -> None:
    """Ablation "\\ TFT" (Sec. 4.4): s_cog fixed to "low fatigue"."""
    from mcga_lm.models.tft import TemporalFusionTransformer

    small_cfg.tft.enabled = False
    tft = TemporalFusionTransformer(small_cfg.perceiver.latent_dim, small_cfg.tft).eval()
    state = tft(torch_mod.randn(4, small_cfg.tft.window, small_cfg.perceiver.latent_dim))
    assert float(state.fatigue.abs().max()) == 0.0
    assert state.level(0) == "low"
    assert float(state.vector.abs().max()) == 0.0


def test_state_depends_on_the_whole_window(torch_mod, small_cfg) -> None:
    from mcga_lm.models.tft import TemporalFusionTransformer

    torch = torch_mod
    torch.manual_seed(0)
    tft = TemporalFusionTransformer(small_cfg.perceiver.latent_dim, small_cfg.tft).eval()
    seq = torch.randn(1, small_cfg.tft.window, small_cfg.perceiver.latent_dim)
    base = tft(seq).vector
    perturbed = seq.clone()
    perturbed[0, 0] += 5.0  # the oldest step in the window
    assert not torch.allclose(base, tft(perturbed).vector, atol=1e-6)


def test_sliding_windows_are_left_padded_and_end_at_the_current_turn(torch_mod) -> None:
    """Sec. 3.3: a window of the last W contextual embeddings."""
    from mcga_lm.data.dataset import sliding_windows

    seq = torch_mod.arange(5, dtype=torch_mod.float32).unsqueeze(1)
    windows = sliding_windows(seq, window=3)
    assert windows.shape == (5, 3, 1)
    assert windows[0].flatten().tolist() == [0.0, 0.0, 0.0]  # left-padded with the first
    assert windows[2].flatten().tolist() == [0.0, 1.0, 2.0]
    assert windows[4].flatten().tolist() == [2.0, 3.0, 4.0]
