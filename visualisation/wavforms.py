import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scienceplots
import matplotlib
import os

plt.style.use(['science', 'no-latex'])

# Output folder
output_dir = r"D:\Uni\Lab\inebriation-voice-detector\plots"
os.makedirs(output_dir, exist_ok=True)

# WAV path
wav_path = r"D:\Uni\Lab\inebriation-voice-detector\data\raw_data\ALC_extended_split\TRAIN\DRUNK\0081008006_h_00.WAV"

# Load audio
sample_rate, data = wavfile.read(wav_path)
if data.ndim == 2:
    data = data.mean(axis=1)

# Time axis
duration = len(data) / sample_rate
time = np.linspace(0, duration, len(data))

# Get viridis colormap colors — first color (deep blue-green)
viridis_colors = plt.cm.viridis(np.linspace(0, 1, 10))
blue_viridis = viridis_colors[3]

# Base filename
base_filename = os.path.splitext(os.path.basename(wav_path))[0]

# === Clean Plot with Viridis Blue ===
fig, ax = plt.subplots(figsize=(20, 6), dpi=300)

ax.plot(time, data, color=blue_viridis, linewidth=1.5)

# Remove axes, ticks, spines, borders
ax.set_axis_off()
plt.margins(x=0, y=0.1)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Save high-res
plt.savefig(os.path.join(output_dir, f"{base_filename}.png"), dpi=600, bbox_inches='tight', pad_inches=0)
plt.savefig(os.path.join(output_dir, f"{base_filename}.pdf"), bbox_inches='tight', pad_inches=0)
plt.close()
