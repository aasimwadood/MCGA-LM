"""Figure reproduction (paper Figs. 2-6).

Every figure is drawn from measured output of THIS implementation. No value from
the paper's tables is hard-coded into a plot. Where the paper's own number is
useful for orientation it goes in the title or an annotation and is labelled as
the paper's, never plotted as if it were ours.

  Fig. 2  confidence calibration curves + ECE
  Fig. 3  per-turn response-latency breakdown
  Fig. 4  ablation study: SACT and hallucination rate
  Fig. 5  simulated fatigue index over a 60-minute conversation
  Fig. 6  cold-start personalisation learning curve with 95% bootstrap CI
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# Non-interactive backend: these scripts run headless.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = {
    "MCGA-LM": "#2a9d8f",
    "RAG-LLM": "#e76f51",
    "M-LLM": "#e9c46a",
    "LLM-Only": "#264653",
    "TouchChat": "#8d99ae",
}


def _save(fig, out_path: str | Path) -> str:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------- Fig 2 #
def plot_calibration(
    curves: Dict[str, Dict[str, Sequence[float]]],
    eces: Dict[str, float],
    out_path: str | Path,
) -> str:
    """Fig. 2: reliability diagram per system, ECE in the legend."""
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for name, curve in curves.items():
        ax.plot(
            curve["confidence"],
            curve["accuracy"],
            marker="o",
            ms=4,
            lw=1.6,
            color=PALETTE.get(name),
            label=f"{name} (ECE={eces.get(name, float('nan')):.2f})",
        )
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Fraction of positives (accuracy)")
    ax.set_title("Confidence calibration curves")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    return _save(fig, out_path)


# --------------------------------------------------------------------- Fig 3 #
def plot_latency_breakdown(
    components: Dict[str, float],
    out_path: str | Path,
    errors: Optional[Dict[str, float]] = None,
    paper_total_ms: float = 457.0,
) -> str:
    """Fig. 3: per-turn response latency, defined switch-press -> display.

    The 2000 ms physiological sliding window is a continuously updated buffer and
    is excluded from the total, exactly as Table 6 states.
    """
    names = list(components)
    values = [components[n] for n in names]
    errs = [(errors or {}).get(n, 0.0) for n in names]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.barh(names, values, xerr=errs, color="#4472c4", height=0.6)
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.02, i, f"{v:.0f} ms", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Response latency contribution (ms)")
    ax.set_title(
        f"Measured per-turn response latency (total {sum(values):.0f} ms)\n"
        f"paper Table 6 reports {paper_total_ms:.0f} ms on a Jetson AGX Orin",
        fontsize=9,
    )
    ax.grid(axis="x", alpha=0.25)
    fig.text(
        0.5,
        -0.04,
        "The 2000 ms physiological sliding window is a continuously updated buffer, "
        "not a per-turn cost, and is excluded (Table 6).",
        ha="center",
        fontsize=7,
        style="italic",
    )
    return _save(fig, out_path)


# --------------------------------------------------------------------- Fig 4 #
def plot_ablation(
    names: Sequence[str],
    sact_mean: Sequence[float],
    sact_sd: Sequence[float],
    halluc_mean: Sequence[float],
    halluc_sd: Sequence[float],
    out_path: str | Path,
) -> str:
    """Fig. 4: SACT and hallucination rate for the full system and each ablation."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(names, sact_mean, yerr=sact_sd, capsize=4, color="#2a9d8f")
    axes[0].set_ylabel("Switch activations per turn (SACT) $\\downarrow$")
    axes[0].set_title("(a) SACT")
    axes[1].bar(names, np.asarray(halluc_mean) * 100, yerr=np.asarray(halluc_sd) * 100, capsize=4, color="#b5446e")
    axes[1].set_ylabel("Hallucination rate (%) $\\downarrow$")
    axes[1].set_title("(b) Hard hallucination rate")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Ablation study on the synthetic persona suite (N = 20 personas)", fontsize=10)
    fig.tight_layout()
    return _save(fig, out_path)


# --------------------------------------------------------------------- Fig 5 #
def plot_fatigue_trajectory(
    minutes: Sequence[float],
    without_adaptation: Sequence[float],
    with_adaptation: Sequence[float],
    out_path: str | Path,
    onset_min: float = 20.0,
) -> str:
    """Fig. 5: simulated fatigue index over a 60-minute conversation."""
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(minutes, without_adaptation, "--", color="#e63946", label="Without adaptation (simulated)")
    ax.plot(minutes, with_adaptation, "-", color="#2a9d8f", label="MCGA-LM (TFT adaptation)")
    ax.axvline(onset_min, color="grey", ls=":", lw=1, label="Adaptation activates")
    ax.set_xlabel("Conversation duration (minutes)")
    ax.set_ylabel("Fatigue index (normalised, 0-1) $\\downarrow$")
    ax.set_title("Simulated fatigue index under TFT adaptation")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, out_path)


# --------------------------------------------------------------------- Fig 6 #
def plot_learning_curve(
    n_utterances: Sequence[int],
    mean_sact: Sequence[float],
    ci_low: Sequence[float],
    ci_high: Sequence[float],
    baseline_sact: float,
    out_path: str | Path,
    full_personalisation_at: Optional[int] = None,
) -> str:
    """Fig. 6: cold-start personalisation learning curve, 95% bootstrap CI."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.fill_between(n_utterances, ci_low, ci_high, alpha=0.25, color="#4a90d9", label="95% CI")
    ax.plot(n_utterances, mean_sact, "-o", ms=4, color="#1f6fb2", label="MCGA-LM (mean)")
    ax.axhline(baseline_sact, color="#8b1e3f", ls="--", lw=1.5, label="RAG-LLM baseline")
    if full_personalisation_at is not None:
        ax.axvline(full_personalisation_at, color="grey", ls=":", lw=1)
        ax.annotate(
            f"~{full_personalisation_at} utterances -> full personalisation",
            xy=(full_personalisation_at, max(mean_sact)),
            fontsize=7,
            xytext=(4, -4),
            textcoords="offset points",
        )
    ax.set_xlabel("Number of accepted utterances")
    ax.set_ylabel("Switch activations per turn (SACT) $\\downarrow$")
    ax.set_title("Cold-start personalisation learning curve")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, out_path)


def bootstrap_ci(
    values: Sequence[float], n_iter: int = 2000, seed: int = 0, alpha: float = 0.05
) -> Tuple[float, float, float]:
    """Mean with a percentile bootstrap CI, used by Fig. 6."""
    rng = np.random.default_rng(seed)
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    draws = np.array([rng.choice(x, x.size, replace=True).mean() for _ in range(n_iter)])
    return (
        float(x.mean()),
        float(np.percentile(draws, 100 * alpha / 2)),
        float(np.percentile(draws, 100 * (1 - alpha / 2))),
    )


# ------------------------------------------------------ Sec. 4.10 (no figure) #
def plot_graph_growth(
    curves: Dict[str, Sequence[Dict[str, float]]],
    out_path: str | Path,
    steady_state: Tuple[int, int] = (200, 500),
) -> str:
    """Intent-graph growth against accepted utterances (Sec. 4.10).

    The paper has no figure for Sec. 4.10 -- it argues the bound in prose. This
    plot exists so the argument can be seen rather than taken on trust: node
    count against accepted utterances, with the paper's 200-500 steady-state
    band shaded, and the marginal new-node rate beneath it. A curve that flattens
    inside the band is the claim holding; one that keeps climbing is not.
    """
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.6, 5.4), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})
    lo, hi = steady_state
    ax.axhspan(lo, hi, color="#2a9d8f", alpha=0.10,
               label=f"paper steady state ({lo}-{hi} nodes)")
    for name, curve in curves.items():
        xs = [c["utterances"] for c in curve]
        ax.plot(xs, [c["n_nodes"] for c in curve], lw=1.4, alpha=0.85, label=name)
        ax2.plot(xs, [c["marginal_new_nodes_per_utterance"] for c in curve], lw=1.2, alpha=0.85)
    ax.set_ylabel("Nodes in the intent graph |V|")
    ax.set_title("Intent-graph growth and steady state (paper Sec. 4.10)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    ax2.set_xlabel("Accepted utterances")
    ax2.set_ylabel("New nodes\nper utterance")
    ax2.grid(alpha=0.25)
    ax2.set_title("Marginal node-creation rate: the sub-linearity claim", fontsize=8)
    fig.tight_layout()
    return _save(fig, out_path)
