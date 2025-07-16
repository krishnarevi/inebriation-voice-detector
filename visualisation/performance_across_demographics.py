import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib as mpl

# --- Scientific Poster Style Setup ---
sns.set_theme(style="whitegrid")
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 22,
    "axes.titlesize": 26,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.dpi": 600,
    "axes.linewidth": 2,
    "lines.linewidth": 3,
    "lines.markersize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# --- Data ---
criteria = ['Sex', 'Sex', 'Region', 'Region']
categories = ['Female', 'Male', 'Bayern', 'Others']
uar_values = [0.7444, 0.7660, 0.7560, 0.7564]

# --- Colors (Pastel Dusty) ---
dusty_colors = ['#B19CD9', '#B19CD9', '#7DA7D9', '#7DA7D9']

# --- Plot ---
fig, ax = plt.subplots(figsize=(14, 10))

# --- Bar Positioning Logic ---
bar_width = 0.35  # Width of each individual bar
group_spacing = 0.8 # Space added between different criteria groups

x_tick_labels_positions = []
current_group_base = 0

# Calculate positions for bars and tick labels
for i in range(len(criteria)):
    if i > 0 and criteria[i] != criteria[i-1]:
        current_group_base += group_spacing # Add spacing for new group
    
    x_tick_labels_positions.append(current_group_base)
    current_group_base += bar_width

bars = ax.bar(x_tick_labels_positions, uar_values, width=bar_width, color=dusty_colors)

# Axes and Labels
ax.set_ylabel('UAR (%)')
ax.set_ylim(0.6, 0.8)

# Set x-ticks to the adjusted positions and labels
ax.set_xticks(x_tick_labels_positions)
ax.set_xticklabels(categories)

# Move category names (e.g., 'Sex', 'Region') above the bars
unique_criteria = []
criteria_indices = []
for i, crit in enumerate(criteria):
    if not unique_criteria or crit != unique_criteria[-1]:
        unique_criteria.append(crit)
        criteria_indices.append([i])
    else:
        criteria_indices[-1].append(i)

for i, crit in enumerate(unique_criteria):
    group_x_coords = [x_tick_labels_positions[j] for j in criteria_indices[i]]
    
    # Calculate the center of the group for the label
    if len(group_x_coords) > 1:
        group_center_x = (group_x_coords[0] + group_x_coords[-1] + bar_width) / 2
    else:
        group_center_x = group_x_coords[0] + bar_width / 2

    ax.text(group_center_x, ax.get_ylim()[1] * 0.98, crit, ha='center', va='bottom', fontsize=20, weight='bold')

# Add percentage values on top of bars
for i, (bar, val) in enumerate(zip(bars, uar_values)):
    ax.annotate(f'{val*100:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 8), # Offset to place text above the bar
                textcoords="offset points",
                ha='center', va='bottom')

# # --- Add Separation Line in the Exact Middle of the Figure ---
# # Calculate the overall x-range covered by the bars
# min_x = x_tick_labels_positions[0]
# max_x = x_tick_labels_positions[-1] + bar_width # End of the last bar

# # Calculate the exact middle of this range
# middle_of_figure_x = (min_x + max_x) / 2

# ax.axvline(middle_of_figure_x, color='gray', linestyle='--', linewidth=1.5, ymin=0.05, ymax=0.95)

# Clean grid
ax.grid(False)

# Tight layout and save
plt.tight_layout()

# --- Save ---
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
filename = "uar_by_demographic_dusty_pastel_with_middle_separator"
plt.savefig(os.path.join(output_dir, f"{filename}.png"))
plt.savefig(os.path.join(output_dir, f"{filename}.pdf"))
plt.close()

print(f"Demographic UAR plots saved in: {os.path.abspath(output_dir)}")