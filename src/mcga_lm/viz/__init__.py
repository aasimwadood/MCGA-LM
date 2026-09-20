"""Figure reproduction (paper Figs. 2-6)."""

from .figures import (
    bootstrap_ci,
    plot_ablation,
    plot_calibration,
    plot_fatigue_trajectory,
    plot_graph_growth,
    plot_latency_breakdown,
    plot_learning_curve,
)

__all__ = [
    "plot_calibration",
    "plot_latency_breakdown",
    "plot_ablation",
    "plot_fatigue_trajectory",
    "plot_learning_curve",
    "plot_graph_growth",
    "bootstrap_ci",
]
