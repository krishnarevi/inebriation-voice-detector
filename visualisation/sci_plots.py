import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib as mpl

# --- Scientific Style Setup ---
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

# --- Parser Function ---
def parse_log_content(log_content):
    epochs, val_losses, val_accuracies, val_uars, val_cms = [], [], [], [], []
    best_val_uar, best_val_uar_epoch, best_val_cm = -1.0, -1, None
    test_cm, test_accuracy, test_uar = None, None, None
    lines = log_content.strip().split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        match_epoch = re.match(r"Epoch (\d+): Validation Loss: ([\d.]+), Accuracy: ([\d.]+), UAR: ([\d.]+)", line)
        if match_epoch:
            epoch = int(match_epoch.group(1))
            val_loss, val_accuracy, val_uar = map(float, match_epoch.groups()[1:])
            epochs.append(epoch)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            val_uars.append(val_uar)
            if i + 1 < len(lines) and "Validation Confusion Matrix:" in lines[i+1]:
                cm_row1 = list(map(int, re.findall(r'\d+', lines[i+2])))
                cm_row2 = list(map(int, re.findall(r'\d+', lines[i+3])))
                if len(cm_row1) == 2 and len(cm_row2) == 2:
                    current_cm = np.array([cm_row1, cm_row2])
                    val_cms.append(current_cm)
                    if val_uar > best_val_uar:
                        best_val_uar, best_val_uar_epoch, best_val_cm = val_uar, epoch, current_cm
                i += 3
        if "--- Running Test Set Evaluation ---" in line:
            j = i + 1
            while j < len(lines):
                test_line = lines[j].strip()
                if "Accuracy:" in test_line:
                    test_accuracy = float(test_line.split(":")[1].strip())
                elif "UAR:" in test_line:
                    test_uar = float(test_line.split(":")[1].strip())
                elif "Confusion Matrix:" in test_line:
                    test_cm_row1 = list(map(int, re.findall(r'\d+', lines[j+1])))
                    test_cm_row2 = list(map(int, re.findall(r'\d+', lines[j+2])))
                    if len(test_cm_row1) == 2 and len(test_cm_row2) == 2:
                        test_cm = np.array([test_cm_row1, test_cm_row2])
                    j += 2
                if "--- Training and Evaluation Complete ---" in test_line:
                    break
                j += 1
            break
        i += 1
    return {
        "epochs": epochs,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "val_uars": val_uars,
        "val_cms": val_cms,
        "best_val_uar_epoch": best_val_uar_epoch,
        "best_val_cm": best_val_cm,
        "test_accuracy": test_accuracy,
        "test_uar": test_uar,
        "test_cm": test_cm
    }

# --- Plotting Functions ---
def plot_metrics_over_epochs(epochs, losses, accuracies, uars, output_dir, title_prefix="Validation"):
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    colors = sns.color_palette("colorblind")
    axes[0].plot(epochs, losses, marker='o', linestyle='-', color=colors[0])
    axes[0].set_title(f'{title_prefix} Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    axes[1].plot(epochs, accuracies, marker='o', linestyle='-', color=colors[1])
    axes[1].set_title(f'{title_prefix} Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True)
    axes[2].plot(epochs, uars, marker='o', linestyle='-', color=colors[2])
    axes[2].set_title(f'{title_prefix} UAR')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('UAR')
    axes[2].grid(True)
    plt.tight_layout()
    base_path = os.path.join(output_dir, f'{title_prefix.lower()}_metrics_over_epochs')
    plt.savefig(base_path + '.png', dpi=300)
    plt.savefig(base_path + '.pdf', format='pdf')
    plt.close(fig)

def plot_confusion_matrix(cm, title, output_path, labels=["Sober (0)", "Drunk (1)"]):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=labels, yticklabels=labels, linewidths=.5, square=True)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(output_path + '.png', dpi=300)
    plt.savefig(output_path + '.pdf', format='pdf')
    plt.close()

def plot_confusion_matrix_advanced(cm, title, output_path, labels=["Sober (0)", "Drunk (1)"]):
    plt.figure(figsize=(10, 8))
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
    labels_ann = np.array([
        [f'{TN}\n(Spec: {specificity:.1%})', f'{FP}\n(FPR: {fpr:.1%})'],
        [f'{FN}\n(FNR: {fnr:.1%})', f'{TP}\n(Sens: {sensitivity:.1%})']
    ])
    sns.heatmap(cm, annot=labels_ann, fmt='s', cmap='Blues', cbar=True,
                xticklabels=labels, yticklabels=labels, linewidths=.5, square=True)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(output_path + '.png', dpi=300)
    plt.savefig(output_path + '.pdf', format='pdf')
    plt.close()

# --- Main Execution ---
log_directory = r'D:\Uni\Lab\model\v5_ext_split'
file_path = os.path.join(log_directory, 'finetune_job.16246516.out')
plots_output_dir = r'D:\Uni\Lab\inebriation-voice-detector\plots'
os.makedirs(plots_output_dir, exist_ok=True)
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")
with open(file_path, 'r') as f:
    outfile_content = f.read()
parsed_data = parse_log_content(outfile_content)
if parsed_data["epochs"]:
    plot_metrics_over_epochs(
        parsed_data["epochs"],
        parsed_data["val_losses"],
        parsed_data["val_accuracies"],
        parsed_data["val_uars"],
        plots_output_dir,
        title_prefix="Validation"
    )
if parsed_data["best_val_cm"] is not None:
    base = os.path.join(plots_output_dir, 'best_val_confusion_matrix')
  #  plot_confusion_matrix(parsed_data["best_val_cm"], f'Confusion Matrix - Best Val UAR (Epoch {parsed_data["best_val_uar_epoch"]})', base)
    plot_confusion_matrix_advanced(parsed_data["best_val_cm"], f'Confusion Matrix - Best Val UAR (Epoch {parsed_data["best_val_uar_epoch"]})', base + '_advanced')
if parsed_data["test_cm"] is not None:
    base = os.path.join(plots_output_dir, 'test_confusion_matrix')
    plot_confusion_matrix_advanced(parsed_data["test_cm"], f'Confusion Matrix - Test (Acc: {parsed_data["test_accuracy"]:.3f}, UAR: {parsed_data["test_uar"]:.3f})', base + '_advanced')

print("\n--- High-Quality Scientific Visualizations Saved ---")
print(f"Location: {plots_output_dir}")
