

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


# ------------------------------------------------------------ effect size -- #
def cohens_ds(x1: Sequence[float], x2: Sequence[float]) -> float:
    """Cohen's d_s (Sec. 4.7): pooled *within-condition* SD, not the SD of the
    difference scores. Reproducible directly from a table of means and SDs."""
    a = np.asarray(x1, dtype=float)
    b = np.asarray(x2, dtype=float)
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    denom = np.sqrt((s1**2 + s2**2) / 2.0)
    if denom == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / denom)


def cohens_ds_from_summary(m1: float, s1: float, m2: float, s2: float) -> float:
    """d_s straight from published means/SDs.

    This is the form that can be recomputed from a table of means and SDs alone.
    Note that the effect sizes printed in the paper's Sec. 5.1 are NOT
    reproducible this way -- see :func:`cohens_dz` and DEVIATION D-09.
    """
    denom = np.sqrt((s1**2 + s2**2) / 2.0)
    return float((m1 - m2) / denom) if denom > 0 else float("nan")


def cohens_dz(x1: Sequence[float], x2: Sequence[float]) -> float:
    """Cohen's ``d_z`` for a paired contrast: mean difference over its own SD.

    Sec. 4.7 says "effect sizes are Cohen's ``d`` for paired contrasts" and
    Sec. 5.1 prints "Cohen's ``d`` (paired, over personas)". ``d_z`` is the
    paired form, and unlike :func:`cohens_ds` it needs the per-persona
    differences: it cannot be recovered from a table of means and SDs, because
    the SD of the differences depends on how the two conditions covary across
    personas. DEVIATION D-09 records why this matters here.
    """
    a = np.asarray(x1, dtype=float)
    b = np.asarray(x2, dtype=float)
    if a.size != b.size:
        raise ValueError("d_z is a paired statistic; both samples must be persona-aligned")
    diff = a - b
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("nan")


def implied_sd_of_differences(mean_1: float, mean_2: float, d_z: float) -> float:
    """SD of the paired differences implied by a published ``d_z``.

    Given a printed ``d_z`` and the two condition means, ``sd_diff = dmean / d_z``.
    This is the only way to interrogate a paired effect size reported without the
    per-subject data, and it is what shows that the Sec. 5.1 figures cannot be
    ``d_s`` (D-09): reproducing them as ``d_z`` requires the MCGA-LM vs M-LLM
    differences to vary about five times less across personas than the
    MCGA-LM vs grid differences.
    """
    return float((mean_1 - mean_2) / d_z) if d_z != 0 else float("nan")


# ----------------------------------------------------------------- ANOVA -- #
@dataclass
class ANOVAResult:
    f_stat: float
    df1: float
    df2: float
    p_value: float
    partial_eta_sq: float
    mauchly_w: float
    mauchly_p: float
    gg_epsilon: float
    corrected: bool

    def as_dict(self) -> Dict[str, float]:
        return {
            "F": self.f_stat,
            "df1": self.df1,
            "df2": self.df2,
            "p": self.p_value,
            "partial_eta_sq": self.partial_eta_sq,
            "mauchly_W": self.mauchly_w,
            "mauchly_p": self.mauchly_p,
            "gg_epsilon": self.gg_epsilon,
            "gg_corrected": float(self.corrected),
        }


def repeated_measures_anova(data: np.ndarray, alpha_sphericity: float = 0.05) -> ANOVAResult:
    """One-way repeated-measures ANOVA (Sec. 4.7).

    ``data``: (n_subjects, n_conditions) -- personas x systems in this paper.
    Applies the Greenhouse-Geisser correction when Mauchly's test rejects.
    """
    from scipy import stats

    x = np.asarray(data, dtype=float)
    n, k = x.shape
    if n < 2 or k < 2:
        raise ValueError("need at least 2 subjects and 2 conditions")

    grand = x.mean()
    ss_cond = n * ((x.mean(axis=0) - grand) ** 2).sum()
    ss_subj = k * ((x.mean(axis=1) - grand) ** 2).sum()
    ss_total = ((x - grand) ** 2).sum()
    ss_error = ss_total - ss_cond - ss_subj

    df_cond, df_error = k - 1, (k - 1) * (n - 1)
    ms_cond = ss_cond / df_cond
    ms_error = ss_error / df_error if df_error > 0 else np.nan
    if ms_error is not None and np.isfinite(ms_error) and ms_error > 1e-12:
        f = ms_cond / ms_error
    elif ss_cond <= 1e-12:
        f = 0.0  # no condition effect and no residual: F is defined as 0, not NaN
    else:
        f = np.inf  # a condition effect with zero residual variance

    w, mauchly_p, epsilon = _mauchly_and_gg(x)
    corrected = bool(np.isfinite(mauchly_p) and mauchly_p < alpha_sphericity and k > 2)
    df1, df2 = (df_cond * epsilon, df_error * epsilon) if corrected else (df_cond, df_error)
    p = float(stats.f.sf(f, df1, df2)) if np.isfinite(f) else float("nan")
    eta = float(ss_cond / (ss_cond + ss_error)) if (ss_cond + ss_error) > 0 else float("nan")
    return ANOVAResult(float(f), float(df1), float(df2), p, eta, float(w), float(mauchly_p), float(epsilon), corrected)


def _mauchly_and_gg(x: np.ndarray) -> Tuple[float, float, float]:
    """Mauchly's W with its chi-square test, and the Greenhouse-Geisser epsilon."""
    from scipy import stats

    n, k = x.shape
    if k < 3:
        return 1.0, 1.0, 1.0
    # Orthonormal contrast basis (Helmert), giving the covariance of contrasts.
    contrasts = np.zeros((k, k - 1))
    for j in range(k - 1):
        contrasts[: j + 1, j] = 1.0 / (j + 1)
        contrasts[j + 1, j] = -1.0
        contrasts[:, j] /= np.linalg.norm(contrasts[:, j])
    cov = np.cov(x, rowvar=False, ddof=1)
    t_cov = contrasts.T @ cov @ contrasts
    p = k - 1
    eigenvalues = np.linalg.eigvalsh(t_cov)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    det = float(np.prod(eigenvalues))
    trace = float(eigenvalues.sum())
    w = det / ((trace / p) ** p) if trace > 0 else np.nan
    df = p * (p + 1) / 2 - 1
    f_corr = 1.0 - (2 * p**2 + p + 2) / (6 * p * (n - 1))
    chi2 = -(n - 1) * f_corr * np.log(w) if np.isfinite(w) and w > 0 else np.nan
    p_value = float(stats.chi2.sf(chi2, df)) if np.isfinite(chi2) else float("nan")
    epsilon = float((trace**2) / (p * (eigenvalues**2).sum())) if (eigenvalues**2).sum() > 0 else 1.0
    epsilon = float(np.clip(epsilon, 1.0 / p, 1.0))
    return float(w), p_value, epsilon


# ------------------------------------------------------- non-parametric -- #
def aligned_rank_transform(data: np.ndarray) -> np.ndarray:
    """ART for a one-way repeated-measures design (Sec. 4.7).

    Aligns each cell by removing all effects but the condition effect, then
    ranks globally. ``data``: (n_subjects, n_conditions).
    """
    x = np.asarray(data, dtype=float)
    grand = x.mean()
    subject_effect = x.mean(axis=1, keepdims=True) - grand
    condition_effect = x.mean(axis=0, keepdims=True) - grand
    residual = x - grand - subject_effect - condition_effect
    aligned = residual + condition_effect  # keep only the effect of interest
    from scipy import stats

    flat_ranks = stats.rankdata(aligned.reshape(-1))
    return flat_ranks.reshape(x.shape)


def pairwise_wilcoxon(
    data: np.ndarray, labels: Sequence[str], reference: int = 0
) -> List[Dict[str, float]]:
    """Wilcoxon signed-rank tests against a reference condition (Sec. 4.7)."""
    from scipy import stats

    x = np.asarray(data, dtype=float)
    out: List[Dict[str, float]] = []
    for j in range(x.shape[1]):
        if j == reference:
            continue
        diff = x[:, reference] - x[:, j]
        if np.allclose(diff, 0):
            stat, p = 0.0, 1.0
        else:
            stat, p = stats.wilcoxon(x[:, reference], x[:, j])
        out.append(
            {
                "comparison": f"{labels[reference]} vs {labels[j]}",
                "statistic": float(stat),
                "p": float(p),
                "d_s": cohens_ds(x[:, reference], x[:, j]),
            }
        )
    return out


def paired_t_tests(data: np.ndarray, labels: Sequence[str], reference: int = 0) -> List[Dict[str, float]]:
    """Post-hoc paired t-tests (Table 8's significance markers)."""
    from scipy import stats

    x = np.asarray(data, dtype=float)
    out: List[Dict[str, float]] = []
    for j in range(x.shape[1]):
        if j == reference:
            continue
        t, p = stats.ttest_rel(x[:, reference], x[:, j])
        out.append(
            {
                "comparison": f"{labels[reference]} vs {labels[j]}",
                "statistic": float(t),
                "p": float(p),
                "d_s": cohens_ds(x[:, reference], x[:, j]),
            }
        )
    return out


# ------------------------------------------------------------ correction -- #
def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> Dict[str, list]:
    """Holm-Bonferroni step-down correction within a metric family (Sec. 4.7)."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return {
        "p_adjusted": adjusted.tolist(),
        "reject": (adjusted < alpha).tolist(),
        "alpha": alpha,
    }


def stars(p: float) -> str:
    """Significance markers as printed in Tables 7-8."""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# ------------------------------------------------------------- bootstrap -- #
def bootstrap_difference(
    a: Sequence[float],
    b: Sequence[float],
    n_iter: int = 10_000,
    seed: int = 0,
    statistic=np.mean,
) -> Dict[str, float]:
    """Bootstrap CI for a difference of statistics (Sec. 4.7, 10,000 iterations)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    observed = float(statistic(x) - statistic(y))
    draws = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        draws[i] = statistic(rng.choice(x, x.size, replace=True)) - statistic(
            rng.choice(y, y.size, replace=True)
        )
    return {
        "observed": observed,
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "p_two_sided": float(2 * min((draws <= 0).mean(), (draws >= 0).mean())),
    }


def kl_from_diagonal(confidence: Sequence[float], accuracy: Sequence[float], eps: float = 1e-9) -> float:
    """KL divergence of a calibration curve from perfect calibration (Sec. 4.7)."""
    c = np.clip(np.asarray(confidence, dtype=float), eps, 1 - eps)
    a = np.clip(np.asarray(accuracy, dtype=float), eps, 1 - eps)
    return float(np.sum(a * np.log(a / c) + (1 - a) * np.log((1 - a) / (1 - c))))


def post_hoc_power(effect_f: float, n: int, k: int, alpha: float = 0.05, rho: float = 0.0) -> float:
    """Post-hoc sensitivity for a one-way repeated-measures design.

    Sec. 3.8 states ">90% power to detect large effects (Cohen's f >= 0.40)" at
    N = 20. The noncentrality parameter used here is ``lambda = f^2 * n * k``,
    i.e. no correction for the within-subject correlation. G*Power's repeated-
    measures option inflates lambda by ``1/(1 - rho)``, which raises power
    further; with rho > 0 this function reproduces the paper's claim, and
    without it the figure is slightly below 0.90. Reported as-is rather than
    tuned -- see D-03 in docs/ASSUMPTIONS.md.
    """
    from scipy import stats

    df1, df2 = k - 1, (k - 1) * (n - 1)
    lam = effect_f**2 * n * k / max(1.0 - rho, 1e-6)
    crit = stats.f.ppf(1 - alpha, df1, df2)
    return float(stats.ncf.sf(crit, df1, df2, lam))


def bootstrap_effect_size(
    x1: Sequence[float],
    x2: Sequence[float],
    n_iter: int = 10_000,
    seed: int = 0,
    paired: bool = True,
) -> Dict[str, float]:
    """Cohen's ``d_s`` with a 95% bootstrap percentile CI (Sec. 5.1).

    Sec. 5.1 reports every effect size as a point estimate with an interval --
    "vs. Grid-based AAC is 2.34 [1.62, 3.06]" -- and states that "all intervals
    are 95% bootstrap percentile intervals (10,000 resamples)". The contrasts
    are paired over personas, so the resampling unit is the persona: each draw
    takes the same resampled personas from both conditions, which preserves the
    within-persona pairing the repeated-measures design depends on. Pass
    ``paired=False`` to resample the two conditions independently.
    """
    a = np.asarray(x1, dtype=float)
    b = np.asarray(x2, dtype=float)
    observed = cohens_ds(a, b)
    if paired and a.size != b.size:
        raise ValueError("paired bootstrap needs equally sized, persona-aligned samples")

    rng = np.random.default_rng(seed)
    draws = np.empty(n_iter, dtype=float)
    n = a.size
    for i in range(n_iter):
        if paired:
            idx = rng.integers(0, n, n)
            draws[i] = cohens_ds(a[idx], b[idx])
        else:
            draws[i] = cohens_ds(rng.choice(a, a.size, replace=True), rng.choice(b, b.size, replace=True))
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return {"d_s": observed, "ci_low": float("nan"), "ci_high": float("nan"), "n_iter": float(n_iter)}
    return {
        "d_s": observed,
        "ci_low": float(np.percentile(finite, 2.5)),
        "ci_high": float(np.percentile(finite, 97.5)),
        "n_iter": float(n_iter),
        "n_finite": float(finite.size),
    }


def format_effect_size(result: Mapping[str, float], nd: int = 2) -> str:
    """Render an effect size the way Sec. 5.1 prints it: ``d [low, high]``."""
    return (
        f"{result['d_s']:.{nd}f} [{result['ci_low']:.{nd}f}, {result['ci_high']:.{nd}f}]"
        if np.isfinite(result.get("ci_low", np.nan))
        else f"{result['d_s']:.{nd}f}"
    )
