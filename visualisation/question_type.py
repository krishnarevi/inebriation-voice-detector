import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import matplotlib as mpl

# --- Style Setup ---
sns.set_theme(style="whitegrid")
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.dpi": 600,
    "axes.linewidth": 1.5,
    "lines.linewidth": 2,
    "lines.markersize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# --- Data ---
data = {
    "Category": [
        "Open question", "Open question",
        "Reading", "Reading", "Reading", "Reading", "Reading",
        "Reading", "Reading", "Reading",
        "Open question"
    ],
    "Type": [
        "Describe an image", "Issue a command",
        "Rapid word list", "Read a command", "Read a license plate",
        "Read a number", "Read a temperature", "Read a tongue twister",
        "Read an address", "Spell a word", "Tell a story"
    ],
    "UAR": [0.80, 0.76, 0.74, 0.68, 0.74, 0.78, 0.75, 0.71, 0.79, 0.78, 0.75]
}
df = pd.DataFrame(data)

# ✅ Sort only by UAR (ascending)
df = df.sort_values("UAR", ascending=False)

# Color mapping
color_map = {
    "Open question": '#B19CD9',
    "Reading": '#7DA7D9'
}
df['Color'] = df['Category'].map(color_map)

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 7))

bars = ax.barh(
    df["Type"],
    df["UAR"],
    color=df["Color"],
    height=0.9  # Minimal spacing between bars
)

# Axes formatting
ax.set_xlabel("UAR", labelpad=10)
ax.set_xlim(0, 1.0)
ax.set_ylabel("")
ax.invert_yaxis()  # Lowest at bottom
ax.grid(False)

# ✅ Legend inside the plot (top-right corner)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, label=cat) for cat, color in color_map.items()]
ax.legend(
    handles=legend_elements,
    frameon=False,
    loc='lower right',
    bbox_to_anchor=(0.99, 0.99)
)

plt.tight_layout()

# Save plots
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "uar_sorted_internal_legend.png"))
plt.savefig(os.path.join(output_dir, "uar_sorted_internal_legend.pdf"))
plt.close()

print(f"Plots saved in: {os.path.abspath(output_dir)}")
