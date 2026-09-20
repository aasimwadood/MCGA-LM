"""Ablation design (paper Sec. 4.4), including the full-factorial reading of D-10."""

from __future__ import annotations


# ------------------------------------------------ D-10: full factorial ----- #
def test_full_factorial_design_has_two_to_the_k_cells() -> None:
    """DEVIATION D-10: Sec. 4.4 says "full-factorial" but describes one-at-a-time.

    The design the first sentence claims has 2^k cells for k components. With
    the paper's five that is 32, against the six conditions Fig. 4 plots.
    """
    from mcga_lm.baselines.variants import FACTORS, full_factorial_design

    cells = full_factorial_design()
    assert len(cells) == 2 ** len(FACTORS) == 32
    assert len({c.name for c in cells}) == 32, "cell names must be distinct"


def test_one_at_a_time_conditions_are_a_subset_of_the_factorial_design() -> None:
    """Fig. 4's five ablations are the single-factor cells of the 2^k design."""
    from mcga_lm.baselines.variants import ABLATIONS, full_factorial_design

    names = {c.name for c in full_factorial_design()}
    for ablation in ABLATIONS:
        assert ablation in names, f"{ablation} should be a single-factor cell"
    assert "MCGA-LM" in names, "the empty cell is the full system"


def test_factorial_cells_compose_their_ablations() -> None:
    """Removing two components must disable both, not just the last one."""
    from mcga_lm.baselines.variants import factorial_variant
    from mcga_lm.config import Config

    cfg = factorial_variant(["GAT", "TFT"]).apply(Config())
    assert not cfg.graph.enabled and not cfg.tft.enabled
    assert cfg.safety.enabled, "an untouched component must survive"


def test_fully_stripped_cell_removes_everything() -> None:
    from mcga_lm.baselines.variants import FACTORS, factorial_variant
    from mcga_lm.config import Config

    variant = factorial_variant(list(FACTORS))
    cfg = variant.apply(Config())
    assert not cfg.graph.enabled and not cfg.tft.enabled and not cfg.safety.enabled
    assert not cfg.perceiver.use_cross_attention
    assert variant.ablate_perceiver


def test_unknown_factor_is_rejected() -> None:
    import pytest as _pytest

    from mcga_lm.baselines.variants import factorial_variant

    with _pytest.raises(KeyError):
        factorial_variant(["Telepathy"])
