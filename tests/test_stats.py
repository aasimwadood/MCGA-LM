"""Statistical analysis plan (paper Sec. 4.7) and the effect sizes of Sec. 5.1."""

from __future__ import annotations

import numpy as np
import pytest

from mcga_lm.eval import stats as S

# Table 7 of the paper: mean +/- SD of SACT across the 20 personas.
TABLE_7_SACT = {
    "TouchChat": (12.2, 2.8),
    "LLM-Only": (8.7, 1.6),
    "RAG-LLM": (5.2, 1.1),
    "MCGA-LM": (3.1, 0.7),
    "M-LLM": (5.8, 1.2),
}
# Cohen's d_s recomputed from Table 7's means and SDs. These are NOT the values
# Sec. 5.1 prints -- see PAPER_PRINTED_D below and DEVIATION D-09.
DS_FROM_TABLE_7 = {"TouchChat": 4.46, "LLM-Only": 4.53, "M-LLM": 2.75, "RAG-LLM": 2.28}

# The effect sizes actually printed in Sec. 5.1, with their 95% bootstrap CIs.
PAPER_PRINTED_D = {
    "TouchChat": (2.34, 1.62, 3.06),
    "LLM-Only": (1.87, 1.21, 2.53),
    "M-LLM": (3.81, 3.23, 5.09),
    "RAG-LLM": (1.05, 0.48, 1.62),
}


@pytest.mark.parametrize("baseline,expected", DS_FROM_TABLE_7.items())
def test_cohens_ds_matches_the_value_table_7_implies(baseline: str, expected: float) -> None:
    """d_s is recomputable from a table of means and SDs; this pins the formula."""
    m1, s1 = TABLE_7_SACT["MCGA-LM"]
    m2, s2 = TABLE_7_SACT[baseline]
    assert S.cohens_ds_from_summary(m2, s2, m1, s1) == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize("baseline", list(PAPER_PRINTED_D))
def test_printed_effect_sizes_are_not_d_s_from_table_7(baseline: str) -> None:
    """DEVIATION D-09: Sec. 5.1's effect sizes are not recomputable from Table 7.

    Sec. 4.7 says effect sizes are "Cohen's d for paired contrasts" and Sec. 5.1
    labels them "Cohen's d (paired, over personas)". Every printed value differs
    from the d_s implied by Table 7's means and SDs, so they must be the paired
    d_z, which depends on the per-persona difference scores and cannot be
    checked without them. This test records the discrepancy rather than
    asserting either value is wrong.
    """
    m1, s1 = TABLE_7_SACT["MCGA-LM"]
    m2, s2 = TABLE_7_SACT[baseline]
    d_s = S.cohens_ds_from_summary(m2, s2, m1, s1)
    printed = PAPER_PRINTED_D[baseline][0]
    assert abs(d_s - printed) > 0.4, (
        f"{baseline}: d_s from Table 7 is {d_s:.2f}, Sec. 5.1 prints {printed:.2f}"
    )


def test_printed_effect_sizes_reorder_the_baselines() -> None:
    """D-09, the part that is not a rounding difference.

    Under d_s the largest effect is against a baseline with a large mean gap.
    Sec. 5.1 instead prints its largest effect (3.81) against M-LLM, whose mean
    SACT (5.8) is the second *closest* to MCGA-LM's 3.1. That ordering is only
    reachable with d_z, and only if the MCGA-LM/M-LLM differences vary far less
    across personas than the others -- about five times less.
    """
    m1, _ = TABLE_7_SACT["MCGA-LM"]
    by_ds = sorted(DS_FROM_TABLE_7, key=lambda k: -DS_FROM_TABLE_7[k])
    by_printed = sorted(PAPER_PRINTED_D, key=lambda k: -PAPER_PRINTED_D[k][0])
    assert by_ds[0] != by_printed[0]
    assert by_printed[0] == "M-LLM"

    implied = {
        k: S.implied_sd_of_differences(TABLE_7_SACT[k][0], m1, PAPER_PRINTED_D[k][0])
        for k in PAPER_PRINTED_D
    }
    assert implied["M-LLM"] == pytest.approx(0.71, abs=0.02)
    assert implied["TouchChat"] / implied["M-LLM"] > 4.0


@pytest.mark.parametrize("baseline", list(PAPER_PRINTED_D))
def test_printed_intervals_are_ordered_and_exclude_the_null(baseline: str) -> None:
    """Sec. 5.1: "all large effects (d > 0.8)", with 95% bootstrap intervals."""
    d, low, high = PAPER_PRINTED_D[baseline]
    assert low < high
    assert low > 0.0, "an interval crossing zero would not support the claim"
    assert d > 0.8


def test_cohens_dz_is_the_paired_form() -> None:
    """d_z divides by the SD of the differences, not by a pooled within-SD."""
    a = np.array([3.0, 3.2, 2.9, 3.1, 3.3])
    b = a + 2.0  # a perfectly constant difference
    assert not np.isfinite(S.cohens_dz(a, b)) or abs(S.cohens_dz(a, b)) > 1e6
    assert np.isfinite(S.cohens_ds(a, b)), "d_s stays finite where d_z blows up"


def test_cohens_dz_requires_aligned_samples() -> None:
    with pytest.raises(ValueError):
        S.cohens_dz(np.zeros(5), np.zeros(4))


def test_implied_sd_of_differences_inverts_d_z() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(3.1, 0.7, 40)
    b = rng.normal(5.2, 1.1, 40)
    d_z = S.cohens_dz(b, a)
    assert S.implied_sd_of_differences(b.mean(), a.mean(), d_z) == pytest.approx(
        (b - a).std(ddof=1)
    )


# ------------------------------------------------- bootstrap effect sizes -- #
def test_bootstrap_effect_size_brackets_the_point_estimate() -> None:
    """Sec. 5.1 reports every d with a 95% bootstrap percentile CI."""
    rng = np.random.default_rng(0)
    a = rng.normal(3.1, 0.7, 20)
    b = rng.normal(12.2, 2.8, 20)
    out = S.bootstrap_effect_size(a, b, n_iter=2000, seed=1)
    assert out["ci_low"] < out["d_s"] < out["ci_high"]
    assert out["d_s"] == pytest.approx(S.cohens_ds(a, b))


def test_bootstrap_effect_size_is_deterministic_under_a_seed() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(3, 1, 20), rng.normal(6, 1, 20)
    first = S.bootstrap_effect_size(a, b, n_iter=500, seed=7)
    second = S.bootstrap_effect_size(a, b, n_iter=500, seed=7)
    assert first == second


def test_paired_bootstrap_requires_aligned_samples() -> None:
    with pytest.raises(ValueError):
        S.bootstrap_effect_size(np.zeros(20), np.zeros(19), n_iter=10)


def test_unpaired_bootstrap_allows_unequal_sizes() -> None:
    rng = np.random.default_rng(0)
    out = S.bootstrap_effect_size(rng.normal(3, 1, 20), rng.normal(6, 1, 15), n_iter=200, paired=False)
    assert np.isfinite(out["d_s"])


def test_format_effect_size_matches_the_papers_notation() -> None:
    """Sec. 5.1 prints "2.34 [1.62, 3.06]"."""
    text = S.format_effect_size({"d_s": 2.34, "ci_low": 1.62, "ci_high": 3.06})
    assert text == "2.34 [1.62, 3.06]"


def test_cohens_ds_matches_the_summary_form_on_raw_data() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(3.1, 0.7, 20)
    b = rng.normal(5.2, 1.1, 20)
    assert S.cohens_ds(a, b) == pytest.approx(
        S.cohens_ds_from_summary(a.mean(), a.std(ddof=1), b.mean(), b.std(ddof=1))
    )


def test_cohens_ds_is_zero_for_identical_conditions() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert S.cohens_ds(x, x) == pytest.approx(0.0)


def test_repeated_measures_anova_against_a_hand_computed_example() -> None:
    # 4 subjects x 3 conditions; SS decomposition worked through by hand.
    data = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 5.0], [4.0, 5.0, 6.0], [3.0, 3.0, 4.0]])
    n, k = data.shape
    grand = data.mean()
    ss_cond = n * ((data.mean(axis=0) - grand) ** 2).sum()
    ss_subj = k * ((data.mean(axis=1) - grand) ** 2).sum()
    ss_err = ((data - grand) ** 2).sum() - ss_cond - ss_subj
    expected_f = (ss_cond / (k - 1)) / (ss_err / ((k - 1) * (n - 1)))
    result = S.repeated_measures_anova(data)
    assert result.f_stat == pytest.approx(expected_f)
    assert 0 < result.p_value < 1
    assert 0 <= result.partial_eta_sq <= 1


def test_anova_finds_no_effect_when_conditions_are_identical() -> None:
    data = np.tile(np.array([[1.0, 2.0, 3.0, 4.0]]).T, (1, 3))
    assert S.repeated_measures_anova(data).f_stat == pytest.approx(0.0, abs=1e-9)


def test_greenhouse_geisser_epsilon_is_bounded() -> None:
    rng = np.random.default_rng(1)
    data = rng.normal(size=(20, 5)) * np.array([1.0, 3.0, 0.5, 2.0, 5.0])
    result = S.repeated_measures_anova(data)
    assert 1.0 / (5 - 1) <= result.gg_epsilon <= 1.0


def test_holm_bonferroni_is_step_down_and_monotone() -> None:
    out = S.holm_bonferroni([0.001, 0.02, 0.03, 0.9])
    adjusted = out["p_adjusted"]
    assert adjusted[0] == pytest.approx(0.004)  # 4 x 0.001
    assert adjusted == sorted(adjusted)  # already in ascending raw order
    assert all(a <= 1.0 for a in adjusted)
    assert out["reject"][0] is True and out["reject"][-1] is False


def test_holm_is_never_more_lenient_than_uncorrected() -> None:
    p = [0.01, 0.04, 0.2]
    assert all(a >= b for a, b in zip(S.holm_bonferroni(p)["p_adjusted"], p))


def test_aligned_rank_transform_preserves_shape_and_condition_ordering() -> None:
    data = np.array([[1.0, 5.0, 9.0], [2.0, 6.0, 10.0], [0.5, 4.0, 8.0]])
    ranks = S.aligned_rank_transform(data)
    assert ranks.shape == data.shape
    means = ranks.mean(axis=0)
    assert means[0] < means[1] < means[2]


def test_bootstrap_difference_brackets_the_observed_gap() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(0.04, 0.01, 60), rng.normal(0.21, 0.05, 60)
    out = S.bootstrap_difference(a, b, n_iter=500, seed=0)
    assert out["ci_low"] <= out["observed"] <= out["ci_high"]
    assert out["observed"] < 0  # ECE of the grounded system is lower


def test_stars_thresholds_match_the_table_footnotes() -> None:
    assert S.stars(0.0005) == "***" and S.stars(0.005) == "**"
    assert S.stars(0.04) == "*" and S.stars(0.4) == "n.s."


def test_post_hoc_power_behaves_monotonically() -> None:
    """Sec. 3.8 claims >90% power for Cohen's f >= 0.40 at N = 20, k = 5.

    Whether that threshold is met depends on the noncentrality convention (see
    D-03): with no within-subject correlation this implementation gives ~0.89,
    and it exceeds 0.90 once rho > 0, as a repeated-measures calculator assumes.
    The properties asserted here are the ones that do not depend on that choice.
    """
    base = S.post_hoc_power(0.40, n=20, k=5)
    assert 0.85 < base < 1.0
    assert S.post_hoc_power(0.40, n=20, k=5, rho=0.3) > base
    assert S.post_hoc_power(0.40, n=20, k=5, rho=0.3) > 0.90
    assert S.post_hoc_power(0.10, n=20, k=5) < base  # smaller effect, less power
    assert S.post_hoc_power(0.40, n=40, k=5) > base  # more personas, more power
