"""Fig. 3: ablation results, read from the run's own output file.

Usage:
    python AblationFromRun.py path/to/runs/ablations/ablations.json

Reads the file written by scripts/run_ablations.py of the released code
({"ablations": {name: {metric: {"mean": .., "sd": ..}}}}) and draws horizontal
bars, so the condition labels cannot overlap. Nothing is typed in by hand.
"""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "ablations.json"
with open(path) as fh:
    run = json.load(fh)
table = run["ablations"]

names = list(table)  # run order: full system first, then each ablation
sact = np.array([table[n]["sact"]["mean"] for n in names])
sact_sd = np.array([table[n]["sact"]["sd"] for n in names])
hall = np.array([table[n]["hallucination_hard"]["mean"] for n in names]) * 100
hall_sd = np.array([table[n]["hallucination_hard"]["sd"] for n in names]) * 100

y = np.arange(len(names))[::-1]  # first entry at the top
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 0.45 * len(names) + 1.4), sharey=True)
ax1.barh(y, sact, xerr=sact_sd, capsize=3, color="#2E86AB", edgecolor="black", linewidth=0.5)
ax1.set_xlabel("Switch activations per turn (SACT) ↓")
ax1.set_title("(a) SACT", fontsize=10)
ax2.barh(y, hall, xerr=hall_sd, capsize=3, color="#A23B72", edgecolor="black", linewidth=0.5)
ax2.set_xlabel("Hard hallucination rate (%) ↓")
ax2.set_title("(b) Hallucination rate", fontsize=10)
ax1.set_yticks(y)
ax1.set_yticklabels(names, fontsize=9)
for ax in (ax1, ax2):
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_xlim(left=0)
condition = "induced fatigue" if run.get("fatigued") else "rested"
fig.suptitle(f"Ablations, 20 synthetic personas ({condition}); error bars: SD across personas",
             fontsize=9)
fig.tight_layout()
fig.savefig("fig_ablation.pdf", bbox_inches="tight")
