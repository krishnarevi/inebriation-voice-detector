import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib as mpl

# --- Your Scientific Style Setup ---
sns.set_theme(style="whitegrid")
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 22,
    "axes.titlesize": 26,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.dpi": 300,
    "axes.linewidth": 2,
    "lines.linewidth": 3,
    "lines.markersize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Data
models = ['Simple CNN', 'ResNet-18', 'Wav2Vec2\nFinetuned']
accuracy = [0.565, 0.626, 0.753]
uar = [0.579, 0.625, 0.757]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 10))

# Dusty pastel colors resembling draw.io palette
dusty_purple = '#B19CD9'  # dusty lavender/purple
dusty_blue = '#7DA7D9'    # muted dusty blue

bars1 = ax.bar(x - width/2, accuracy, width, label='Accuracy', color=dusty_purple)
bars2 = ax.bar(x + width/2, uar, width, label='UAR', color=dusty_blue)

ax.set_ylabel('Score (%)')
ax.set_xlabel('Model')
#ax.set_title('Baseline Model Performance')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 1.0)
ax.legend(frameon=False)

# Remove grid
ax.grid(False)

# Annotate bars with percentages
def annotate_percent(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height*100:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 8),  # vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

annotate_percent(bars1)
annotate_percent(bars2)

plt.tight_layout()

# Save plots
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "baseline_model_comparison_dusty_pastel_no_grid.png"))
plt.savefig(os.path.join(output_dir, "baseline_model_comparison_dusty_pastel_no_grid.pdf"))
plt.close()

print(f"Plots saved in: {os.path.abspath(output_dir)}")
