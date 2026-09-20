"""Training objectives (paper Sec. 3.10, Eqs. 9-13)."""

from __future__ import annotations

import math

import pytest


def test_fatigue_loss_is_mse(torch_mod) -> None:
    """Eq. (10): L_fatigue = (1/T) sum (f_hat - f)^2."""
    from mcga_lm.losses import fatigue_loss

    pred = torch_mod.tensor([0.2, 0.5, 0.9])
    target = torch_mod.tensor([0.0, 0.5, 0.5])
    expected = (0.2**2 + 0.0**2 + 0.4**2) / 3
    assert float(fatigue_loss(pred, target)) == pytest.approx(expected, abs=1e-6)


def test_contrastive_loss_is_minimal_when_pairs_align(torch_mod) -> None:
    """Eq. (11): InfoNCE over in-batch negatives, tau_c = 0.07."""
    from mcga_lm.losses import contrastive_loss

    torch = torch_mod
    z = torch.eye(4)
    aligned = float(contrastive_loss(z, z, temperature=0.07))
    shuffled = float(contrastive_loss(z, z.flip(0), temperature=0.07))
    assert aligned < shuffled
    assert aligned >= 0.0


def test_contrastive_loss_hits_the_analytic_value_for_orthogonal_pairs(torch_mod) -> None:
    from mcga_lm.losses import contrastive_loss

    torch = torch_mod
    z = torch.eye(3)
    # cosine sims: 1 on the diagonal, 0 off it -> -log(e^{1/t} / (e^{1/t} + 2))
    t = 0.07
    expected = -math.log(math.exp(1 / t) / (math.exp(1 / t) + 2 * math.exp(0.0)))
    assert float(contrastive_loss(z, z, temperature=t)) == pytest.approx(expected, abs=1e-5)


def test_contrastive_loss_degenerates_safely_for_a_single_example(torch_mod) -> None:
    from mcga_lm.losses import contrastive_loss

    z = torch_mod.randn(1, 8)
    assert float(contrastive_loss(z, z)) == 0.0


def test_reconstruction_loss_is_zero_on_a_perfect_reconstruction(torch_mod) -> None:
    """Eq. (12)."""
    from mcga_lm.losses import reconstruction_loss

    x = torch_mod.randn(2, 5, 3)
    assert float(reconstruction_loss(x, x)) == pytest.approx(0.0, abs=1e-9)
    assert float(reconstruction_loss(x, x + 1.0)) == pytest.approx(1.0, abs=1e-5)


def test_intent_loss_is_cross_entropy_over_attention(torch_mod) -> None:
    """Eq. (13): L_intent = - sum_i y_i log y_hat_i."""
    from mcga_lm.losses import intent_loss

    torch = torch_mod
    scores = torch.tensor([[0.7, 0.2, 0.1]])
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    assert float(intent_loss(scores, targets)) == pytest.approx(-math.log(0.7), abs=1e-6)


def test_intent_loss_falls_as_the_right_node_gains_attention(torch_mod) -> None:
    from mcga_lm.losses import intent_loss

    torch = torch_mod
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    good = float(intent_loss(torch.tensor([[0.9, 0.05, 0.05]]), targets))
    bad = float(intent_loss(torch.tensor([[0.1, 0.45, 0.45]]), targets))
    assert good < bad


def test_intent_loss_normalises_by_the_number_of_positives(torch_mod) -> None:
    """A-13: examples with differently sized intent sets contribute comparably."""
    from mcga_lm.losses import intent_loss

    torch = torch_mod
    scores = torch.tensor([[0.5, 0.5, 0.0]])
    one = float(intent_loss(scores, torch.tensor([[1.0, 0.0, 0.0]])))
    two = float(intent_loss(scores, torch.tensor([[1.0, 1.0, 0.0]])))
    assert one == pytest.approx(two, abs=1e-6)


def test_total_loss_applies_the_lambdas_of_equation_9(torch_mod, cfg) -> None:
    """Eq. (9): L = L_fat + 0.5 L_con + 0.3 L_rec + 1.0 L_int."""
    from mcga_lm.losses import total_loss

    torch = torch_mod
    x = torch.randn(4, 3, 2)
    breakdown = total_loss(
        cfg.loss,
        fatigue_pred=torch.zeros(4), fatigue_target=torch.ones(4),
        z_ctx=torch.eye(4), z_utt=torch.eye(4),
        x_phys=x, x_recon=x + 1.0,
        node_scores=torch.full((4, 3), 1 / 3), intent_targets=torch.eye(4, 3),
    )
    expected = (
        float(breakdown.fatigue)
        + 0.5 * float(breakdown.contrastive)
        + 0.3 * float(breakdown.recon)
        + 1.0 * float(breakdown.intent)
    )
    assert float(breakdown.total) == pytest.approx(expected, abs=1e-6)


def test_missing_terms_contribute_zero(torch_mod, cfg) -> None:
    """Sec. 3.10: L_fatigue "is not used" in the synthetic evaluation."""
    from mcga_lm.losses import total_loss

    torch = torch_mod
    breakdown = total_loss(cfg.loss, z_ctx=torch.eye(3), z_utt=torch.eye(3))
    assert float(breakdown.fatigue) == 0.0
    assert float(breakdown.recon) == 0.0
    assert float(breakdown.intent) == 0.0
    assert float(breakdown.total) == pytest.approx(0.5 * float(breakdown.contrastive), abs=1e-6)
