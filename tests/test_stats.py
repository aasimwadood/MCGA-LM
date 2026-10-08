"""Statistical analysis plan (paper Sec. 4.8) and the effect sizes of Sec. 5.1."""

from __future__ import annotations

import numpy as np
import pytest

from mcga_lm.eval import stats as S

# Table 6 of the paper: mean +/- SD of SACT across the 20 personas.
TABLE_6_SACT = {
    "TouchChat": (12.2, 2.8),
    "LLM-Only": (8.7, 1.6),
    "RAG-LLM": (5.2, 1.1),
    "MCGA-LM": (3.1, 0.7),
    "M-LLM": (5.8, 1.2),
}
# Cohen's d_s recomputed from Table 6's means and SDs: the values the revised
# Sec. 5.1 reports. They are NOT the values the original Sec. 5.1 printed -- see
# ORIGINAL_PRINTED_D below and ERRATA E-1.
DS_FROM_TABLE_6 = {"TouchChat": 4.46, "LLM-Only": 4.53, "M-LLM": 2.75, "RAG-LLM": 2.28}

# The effect sizes the original Sec. 5.1 printed, with their 95% bootstrap CIs.
# Replaced by d_s in the revision: no per-persona data for them exists (ERRATA E-1).
ORIGINAL_PRINTED_D = {
    "TouchChat": (2.34, 1.62, 3.06),
    "LLM-Only": (1.87, 1.21, 2.53),
    "M-LLM": (3.81, 3.23, 5.09),
    "RAG-LLM": (1.05, 0.48, 1.62),
}


@pytest.mark.parametrize("baseline,expected", DS_FROM_TABLE_6.items())
def test_cohens_ds_matches_the_value_table_6_implies(baseline: str, expected: float) -> None:
    """d_s is recomputable from a table of means and SDs; this pins the formula
    and the four values the revised Sec. 5.1 reports."""
    m1, s1 = TABLE_6_SACT["MCGA-LM"]
    m2, s2 = TABLE_6_SACT[baseline]
    assert S.cohens_ds_from_summary(m2, s2, m1, s1) == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize("baseline", list(ORIGINAL_PRINTED_D))
def test_printed_effect_sizes_are_not_d_s_from_table_6(baseline: str) -> None:
    """ERRATA E-1: the original Sec. 5.1's effect sizes are not recomputable from Table 6.

    The original Sec. 4.7 said effect sizes are "Cohen's d for paired contrasts"
    and Sec. 5.1 labelled them "Cohen's d (paired, over personas)". Every printed
    value differs from the d_s implied by Table 6's means and SDs, so they are
    consistent only with the paired d_z, which needs per-persona difference
    scores that were never published and have not been found. This test records
    the discrepancy that led the revision to withdraw them.
    """
    m1, s1 = TABLE_6_SACT["MCGA-LM"]
    m2, s2 = TABLE_6_SACT[baseline]
    d_s = S.cohens_ds_from_summary(m2, s2, m1, s1)
    printed = ORIGINAL_PRINTED_D[baseline][0]
    assert abs(d_s - printed) > 0.4, (
        f"{baseline}: d_s from Table 6 is {d_s:.2f}, Sec. 5.1 prints {printed:.2f}"
    )


def test_printed_effect_sizes_reorder_the_baselines() -> None:
    """ERRATA E-1, the part that is not a rounding difference.

    Under d_s the largest effect is against a baseline with a large mean gap.
    The original Sec. 5.1 instead printed its largest effect (3.81) against M-LLM, whose mean
    SACT (5.8) is the second *closest* to MCGA-LM's 3.1. That ordering is only
    reachable with d_z, and only if the MCGA-LM/M-LLM differences vary far less
    across personas than the others -- about five times less.
    """
    m1, _ = TABLE_6_SACT["MCGA-LM"]
    by_ds = sorted(DS_FROM_TABLE_6, key=lambda k: -DS_FROM_TABLE_6[k])
    by_printed = sorted(ORIGINAL_PRINTED_D, key=lambda k: -ORIGINAL_PRINTED_D[k][0])
    assert by_ds[0] != by_printed[0]
    assert by_printed[0] == "M-LLM"

    implied = {
        k: S.implied_sd_of_differences(TABLE_6_SACT[k][0], m1, ORIGINAL_PRINTED_D[k][0])
        for k in ORIGINAL_PRINTED_D
    }
    assert implied["M-LLM"] == pytest.approx(0.71, abs=0.02)
    assert implied["TouchChat"] / implied["M-LLM"] > 4.0


@pytest.mark.parametrize("baseline", list(DS_FROM_TABLE_6))
def test_revised_effect_sizes_stay_large_at_the_rounding_extremes(baseline: str) -> None:
    """The revised Sec. 5.1 keeps "all large effects (d > 0.8)", now as d_s.

    Table 6 rounds to one decimal place, so each mean and SD may be off by up to
    0.05. The claim has to survive the least favourable combination: the
    smallest mean gap with the largest SDs.
    """
    m1, s1 = TABLE_6_SACT["MCGA-LM"]
    m2, s2 = TABLE_6_SACT[baseline]
    h = 0.05
    assert S.cohens_ds_from_summary(m2 - h, s2 + h, m1 + h, s1 + h) > 0.8


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
    """The original Sec. 5.1 reported every d with a 95% bootstrap percentile CI."""
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


def test_format_effect_size_matches_the_original_notation() -> None:
    """The original Sec. 5.1 printed "2.34 [1.62, 3.06]"."""
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


def test_stars_thresholds_match_the_original_table_footnotes() -> None:
    assert S.stars(0.0005) == "***" and S.stars(0.005) == "**"
    assert S.stars(0.04) == "*" and S.stars(0.4) == "n.s."


def test_post_hoc_power_behaves_monotonically() -> None:
    """Sec. 3.8 claims >90% power for Cohen's f >= 0.40 at N = 20, k = 5.

    Whether that threshold is met depends on the noncentrality convention (see
    ``post_hoc_power``): with no within-subject correlation this implementation gives ~0.89,
    and it exceeds 0.90 once rho > 0, as a repeated-measures calculator assumes.
    The properties asserted here are the ones that do not depend on that choice.
    """
    base = S.post_hoc_power(0.40, n=20, k=5)
    assert 0.85 < base < 1.0
    assert S.post_hoc_power(0.40, n=20, k=5, rho=0.3) > base
    assert S.post_hoc_power(0.40, n=20, k=5, rho=0.3) > 0.90
    assert S.post_hoc_power(0.10, n=20, k=5) < base  # smaller effect, less power
    assert S.post_hoc_power(0.40, n=40, k=5) > base  # more personas, more power


# ------------------------------------------------------------ calibration -- #
def test_kl_from_diagonal_is_zero_for_a_calibrated_curve() -> None:
    assert S.kl_from_diagonal([0.2, 0.5, 0.9], [0.2, 0.5, 0.9]) == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(S.kl_from_diagonal([], []))


def test_count_weights_stop_a_one_turn_bin_dominating_the_kl() -> None:
    confidence, accuracy, counts = [0.5, 0.95], [0.5, 0.0], [99, 1]
    unweighted = S.kl_from_diagonal(confidence, accuracy)
    weighted = S.kl_from_diagonal(confidence, accuracy, weights=counts)
    assert weighted == pytest.approx(unweighted / 100)


def test_paired_bootstrap_difference_keeps_personas_together() -> None:
    """A constant per-persona gap has no paired sampling variability at all."""
    b = np.random.default_rng(0).normal(0.2, 0.05, 20)
    out = S.bootstrap_difference(b + 0.1, b, n_iter=300, seed=0, paired=True)
    assert out["ci_low"] == pytest.approx(0.1) and out["ci_high"] == pytest.approx(0.1)
    assert out["p_two_sided"] == pytest.approx(2 / 301)  # never zero: bounded by the resample count
    with pytest.raises(ValueError):
        S.bootstrap_difference(np.zeros(20), np.zeros(19), n_iter=10, paired=True)


def test_holm_leaves_an_undefined_p_undefined() -> None:
    tests = S._holm([{"p": 0.01}, {"p": float("nan")}, {"p": 0.02}], alpha=0.05)
    assert tests[0]["p_holm"] == pytest.approx(0.02) and tests[2]["p_holm"] == pytest.approx(0.02)
    assert np.isnan(tests[1]["p_holm"]) and tests[1]["stars"] == "" and not tests[1]["significant"]


# ---------------------------------------------------------- analysis plan -- #
# Table 7 of the paper: mean +/- SD per generative system.
TABLE_7 = {
    "MCGA-LM": {"ihr@1": (0.68, 0.07), "ihr@5": (0.96, 0.03), "bleu4": (0.52, 0.07), "rouge_l": (0.68, 0.06), "ece": (0.04, 0.02)},
    "RAG-LLM": {"ihr@1": (0.54, 0.06), "ihr@5": (0.82, 0.06), "bleu4": (0.45, 0.07), "rouge_l": (0.60, 0.07), "ece": (0.09, 0.03)},
    "M-LLM": {"ihr@1": (0.41, 0.07), "ihr@5": (0.69, 0.07), "bleu4": (0.39, 0.08), "rouge_l": (0.53, 0.06), "ece": (0.15, 0.04)},
    "LLM-Only": {"ihr@1": (0.28, 0.06), "ihr@5": (0.55, 0.08), "bleu4": (0.31, 0.06), "rouge_l": (0.45, 0.07), "ece": (0.21, 0.05)},
}
TABLE_6_IHR3_HAL = {
    "MCGA-LM": ((0.89, 0.05), (0.02, 0.01)),
    "RAG-LLM": ((0.71, 0.07), (0.07, 0.03)),
    "M-LLM": ((0.54, 0.06), (0.09, 0.02)),
    "LLM-Only": ((0.42, 0.09), (0.18, 0.05)),
}


def _simulated_plan_inputs(n: int = 20, seed: int = 0):
    """Per-persona scores drawn around the paper's Tables 6 and 7, with a shared
    persona effect so the conditions are correlated, as in a repeated-measures design."""
    rng = np.random.default_rng(seed)
    persona = rng.normal(0, 1, n)
    sact = {s: m + 0.5 * sd * persona + sd * rng.normal(0, 0.87, n) for s, (m, sd) in TABLE_6_SACT.items()}
    metrics = {}
    for system, cols in TABLE_7.items():
        metrics[system] = {k: m + sd * rng.normal(0, 1, n) for k, (m, sd) in cols.items()}
        (ihr, ihr_sd), (hal, hal_sd) = TABLE_6_IHR3_HAL[system]
        metrics[system]["ihr@3"] = ihr + ihr_sd * rng.normal(0, 1, n)
        metrics[system]["hallucination_hard"] = np.abs(hal + hal_sd * rng.normal(0, 1, n))
        metrics[system]["calibration_kl"] = metrics[system]["ece"] ** 2 * 5
    metrics["LLM-Only"]["calibration_kl"][3] = np.nan  # one persona with no offered turns
    return sact, metrics


def test_analysis_plan_runs_every_step_of_sec_4_8() -> None:
    sact, metrics = _simulated_plan_inputs()
    plan = S.analysis_plan(sact, metrics, n_iter=300, seed=0)

    assert plan["sact_anova"]["systems"] == list(TABLE_6_SACT)
    assert plan["sact_anova"]["p"] < 0.001
    assert [c["system"] for c in plan["sact_pairwise"]] == ["TouchChat", "LLM-Only", "RAG-LLM", "M-LLM"]
    assert all(c["stars"] == "***" for c in plan["sact_pairwise"])

    assert set(plan["nonparametric"]) == set(S.NONPARAMETRIC_METRICS)
    for res in plan["nonparametric"].values():
        assert res["art_anova"]["p"] < 0.05
        assert len(res["pairwise_wilcoxon"]) == 3

    # Table 7: three baselines per column, each with a Holm-corrected marker.
    assert set(plan["table_7"]) == set(S.TABLE_7_METRICS)
    for tests in plan["table_7"].values():
        assert [t["system"] for t in tests] == ["RAG-LLM", "M-LLM", "LLM-Only"]
        assert all(t["p_holm"] >= t["p"] for t in tests)
        assert all(t["significant"] and t["stars"] for t in tests)

    calibration = plan["calibration"]
    assert "LLM-Only" not in calibration["kl_from_diagonal"]  # a persona's KL is undefined
    for c in calibration["ece_bootstrap"]:
        assert c["ci_low"] <= c["ece_difference"] <= c["ci_high"]
        assert c["ece_difference"] < 0  # MCGA-LM is the best calibrated, as in Table 7


def test_analysis_plan_needs_the_reference_system() -> None:
    sact, metrics = _simulated_plan_inputs()
    sact.pop("MCGA-LM")
    plan = S.analysis_plan(sact, {}, n_iter=10)
    assert "sact_anova" not in plan
    assert plan["table_7"] == {} and plan["nonparametric"] == {} and plan["calibration"] == {}
