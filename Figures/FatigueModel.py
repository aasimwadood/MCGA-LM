"""Fig. 4: the simulator's fatigue model (Section 4.5), plotted without noise.

This plots the model the simulator uses, not measured data. It mirrors
FatigueModel.value / adapted_value in src/mcga_lm/data/physiology.py of the
released code:

    f(t) = f0 + (f_max - f0) * (1 - 2^(-t / t_half))
    adapted: after onset, each further increment is reduced by `relief`.
"""
import numpy as np
import matplotlib.pyplot as plt

F0, F_MAX, T_HALF = 0.05, 0.8, 30.0  # baseline, peak, half-life (min)
RELIEF, ONSET = 0.25, 20.0           # assumed adaptation effect (ASSUMPTION A-15)


def fatigue(t):
    return F0 + (F_MAX - F0) * (1.0 - np.power(0.5, t / T_HALF))


def fatigue_adapted(t):
    base = fatigue(t)
    before = fatigue(np.minimum(t, ONSET))
    return np.where(t <= ONSET, base, before + (base - before) * (1.0 - RELIEF))


t = np.linspace(0, 60, 241)
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(t, fatigue(t), "--", color="#C0392B", linewidth=2, label="Non-adaptive (model)")
ax.plot(t, fatigue_adapted(t), "-", color="#1E8449", linewidth=2.2,
        label=f"Adaptive (model; assumed {int(RELIEF * 100)}% lower increment)")
ax.axvline(ONSET, color="gray", linestyle=":", linewidth=1.2)
ax.text(ONSET + 0.8, 0.06, "adaptation onset\n(assumed)", fontsize=8, color="dimgray")
ax.set_xlabel("Conversation duration (minutes)")
ax.set_ylabel("Simulated fatigue index (0–1)")
ax.set_xlim(0, 60)
ax.set_ylim(0, 1)
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_title("Simulator fatigue model (assumption, not a measured effect)", fontsize=10)
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig("fig_fatigue_over_time.pdf", bbox_inches="tight")
