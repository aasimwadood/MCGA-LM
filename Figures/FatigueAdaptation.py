systems = ['MCGA-LM', 'RAG-LLM', 'LLM-Only']
rested_wpm = [24.7, 18.2, 12.5]
fatigued_wpm = [22.1, 13.6, 9.8]
rested_std = [3.0, 2.5, 2.3]
fatigued_std = [2.8, 2.4, 2.0]

x = np.arange(len(systems))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, rested_wpm, width, yerr=rested_std, capsize=5,
       label='Rested', color='#2E86AB', edgecolor='black')
ax.bar(x + width/2, fatigued_wpm, width, yerr=fatigued_std, capsize=5,
       label='Fatigued', color='#A23B72', edgecolor='black')
ax.set_ylabel('Words Per Minute (WPM) ↑', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(systems, fontsize=11)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_title('Communication Rate Under Induced Fatigue', fontweight='bold')
plt.tight_layout()
plt.savefig('fig_fatigue_wpm.pdf', dpi=300)
plt.show()