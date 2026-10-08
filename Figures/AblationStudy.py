import numpy as np
import matplotlib.pyplot as plt

# Data from paper: SACT (mean ± std) and Hallucination % (mean ± std)
systems = ['Full MCGA-LM', '\\ Perceiver IO', '\\ TFT (no fatigue)', 
           '\\ GAT (no memory)', '\\ Bayesian Gate']
sact_mean = [3.1, 3.8, 4.2, 5.8, 3.5]
sact_std  = [0.7, 0.9, 1.0, 1.2, 0.8]
hall_mean = [2.0, 4.0, 3.0, 9.0, 11.0]
hall_std  = [1.0, 1.5, 1.2, 2.0, 2.5]

x = np.arange(len(systems))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
fig.patch.set_facecolor('white')

# SACT
ax1.bar(x - width/2, sact_mean, width, yerr=sact_std, capsize=5,
        color='#2E86AB', edgecolor='black')
ax1.set_ylabel('Switch Activations per Turn (SACT) ↓', fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(systems, rotation=45, ha='right', fontsize=9)
ax1.set_ylim(0, 7)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.set_title('(a) SACT', fontweight='bold')

# Hallucination rate
ax2.bar(x + width/2, hall_mean, width, yerr=hall_std, capsize=5,
        color='#A23B72', edgecolor='black')
ax2.set_ylabel('Hallucination Rate (%) ↓', fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(systems, rotation=45, ha='right', fontsize=9)
ax2.set_ylim(0, 15)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.set_title('(b) Hallucination Rate', fontweight='bold')

plt.tight_layout()
plt.savefig('fig_ablation.pdf', dpi=300, bbox_inches='tight')
plt.show()