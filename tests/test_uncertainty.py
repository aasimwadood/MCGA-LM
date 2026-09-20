"""MC-Dropout uncertainty gate (paper Sec. 3.6, Eq. 8)."""

from __future__ import annotations

import pytest


def test_mc_dropout_returns_nonnegative_variance_and_bounded_confidence(torch_mod) -> None:
    from mcga_lm.models.intent_head import IntentScoringHead

    torch = torch_mod
    torch.manual_seed(0)
    head = IntentScoringHead(context_dim=16, token_dim=8, hidden=32, dropout=0.2)
    estimate = head.mc_dropout(torch.randn(3, 16), torch.randn(3, 6, 8), n_passes=20)
    assert estimate.variance.shape == (3,)
    assert bool((estimate.variance >= 0).all())
    assert bool(((estimate.confidence >= 0) & (estimate.confidence <= 1)).all())
    assert estimate.n_passes == 20


def test_variance_is_zero_without_dropout(torch_mod) -> None:
    """Eq. (8) measures spread across sampled weight sets w^(n); with p = 0 there
    is no spread."""
    from mcga_lm.models.intent_head import IntentScoringHead

    torch = torch_mod
    head = IntentScoringHead(context_dim=16, token_dim=8, hidden=32, dropout=0.0)
    estimate = head.mc_dropout(torch.randn(2, 16), torch.randn(2, 5, 8), n_passes=10)
    assert float(estimate.variance.max()) == pytest.approx(0.0, abs=1e-12)


def test_more_dropout_means_more_uncertainty(torch_mod) -> None:
    from mcga_lm.models.intent_head import IntentScoringHead

    torch = torch_mod
    torch.manual_seed(0)
    context, tokens = torch.randn(8, 16), torch.randn(8, 6, 8)
    low = IntentScoringHead(16, 8, hidden=32, dropout=0.05)
    high = IntentScoringHead(16, 8, hidden=32, dropout=0.5)
    high.load_state_dict(low.state_dict())
    v_low = float(low.mc_dropout(context, tokens, n_passes=40).variance.mean())
    v_high = float(high.mc_dropout(context, tokens, n_passes=40).variance.mean())
    assert v_high > v_low


def test_variance_is_a_token_mean(torch_mod) -> None:
    """Eq. (8): Var(y_hat) = (1/|y_hat|) sum_l Var(...)."""
    from mcga_lm.models.intent_head import IntentScoringHead

    torch = torch_mod
    torch.manual_seed(0)
    head = IntentScoringHead(16, 8, hidden=32, dropout=0.2)
    est = head.mc_dropout(torch.randn(2, 16), torch.randn(2, 7, 8), n_passes=12)
    assert torch.allclose(est.variance, est.per_token_var.mean(dim=-1), atol=1e-7)


def test_padding_mask_excludes_padded_tokens(torch_mod) -> None:
    from mcga_lm.models.intent_head import IntentScoringHead

    torch = torch_mod
    torch.manual_seed(0)
    head = IntentScoringHead(16, 8, hidden=32, dropout=0.2)
    context, tokens = torch.randn(1, 16), torch.randn(1, 6, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0]])
    masked = head.mc_dropout(context, tokens, n_passes=16, mask=mask)
    assert torch.allclose(masked.variance, masked.per_token_var[:, :3].mean(dim=-1), atol=1e-7)


def test_head_is_left_in_eval_mode_after_scoring(torch_mod) -> None:
    """MC Dropout flips dropout on temporarily; it must not leak into training state."""
    from mcga_lm.models.intent_head import IntentScoringHead

    head = IntentScoringHead(16, 8, hidden=32, dropout=0.2).eval()
    head.mc_dropout(torch_mod.randn(1, 16), torch_mod.randn(1, 4, 8), n_passes=4)
    assert head.training is False


def test_reconstruction_decoder_shape_matches_x_phys(torch_mod, small_cfg) -> None:
    """Eq. (12): the decoder reconstructs x_phys from the latent array Z."""
    from mcga_lm.models.intent_head import PhysiologyDecoder

    decoder = PhysiologyDecoder(
        latent_dim=small_cfg.perceiver.latent_dim,
        out_steps=small_cfg.inputs.phys_window,
        out_features=small_cfg.inputs.phys_dim,
    )
    latents = torch_mod.randn(3, small_cfg.perceiver.num_latents, small_cfg.perceiver.latent_dim)
    out = decoder(latents)
    assert out.shape == (3, small_cfg.inputs.phys_window, small_cfg.inputs.phys_dim)
