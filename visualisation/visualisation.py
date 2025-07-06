import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os 

def parse_log_content(log_content):
    epochs = []
    val_losses = []
    val_accuracies = []
    val_uars = []
    val_cms = []
    
    best_val_uar = -1.0
    best_val_uar_epoch = -1
    best_val_cm = None

    test_cm = None
    test_accuracy = None
    test_uar = None

    lines = log_content.strip().split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Match Epoch Validation Metrics
        match_epoch = re.match(r"Epoch (\d+): Validation Loss: ([\d.]+), Accuracy: ([\d.]+), UAR: ([\d.]+)", line)
        if match_epoch:
            epoch = int(match_epoch.group(1))
            val_loss = float(match_epoch.group(2))
            val_accuracy = float(match_epoch.group(3))
            val_uar = float(match_epoch.group(4))

            epochs.append(epoch)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            val_uars.append(val_uar)

            # Read Confusion Matrix
            # Expecting "Validation Confusion Matrix:" followed by two lines for the matrix
            if i + 1 < len(lines) and "Validation Confusion Matrix:" in lines[i+1]:
                cm_row1_str = lines[i+2].strip().replace('[', '').replace(']', '')
                cm_row2_str = lines[i+3].strip().replace('[', '').replace(']', '')
                
                cm_row1 = list(map(int, re.findall(r'\d+', cm_row1_str))) 
                cm_row2 = list(map(int, re.findall(r'\d+', cm_row2_str))) 
                
                if len(cm_row1) == 2 and len(cm_row2) == 2: 
                    current_cm = np.array([cm_row1, cm_row2])
                    val_cms.append(current_cm)

                    # Update best validation UAR and its CM
                    if val_uar > best_val_uar:
                        best_val_uar = val_uar
                        best_val_uar_epoch = epoch
                        best_val_cm = current_cm
                else:
                    print(f"Warning: Could not parse confusion matrix for Epoch {epoch}. Skipping.")

                i += 3 
        
        # Match Test Results
        if "--- Running Test Set Evaluation ---" in line:
            # Find Test Results section
            j = i + 1
            while j < len(lines):
                test_line = lines[j].strip()
                if "Accuracy:" in test_line:
                    test_accuracy = float(test_line.split(":")[1].strip())
                elif "UAR:" in test_line:
                    test_uar = float(test_line.split(":")[1].strip())
                elif "Confusion Matrix:" in test_line:
                    cm_row1_str = lines[j+1].strip().replace('[', '').replace(']', '')
                    cm_row2_str = lines[j+2].strip().replace('[', '').replace(']', '')
                    
                    test_cm_row1 = list(map(int, re.findall(r'\d+', cm_row1_str)))
                    test_cm_row2 = list(map(int, re.findall(r'\d+', cm_row2_str)))

                    if len(test_cm_row1) == 2 and len(test_cm_row2) == 2: 
                        test_cm = np.array([test_cm_row1, test_cm_row2])
                    else:
                        print("Warning: Could not parse test confusion matrix. Skipping.")
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

def plot_metrics_over_epochs(epochs, losses, accuracies, uars, output_dir, title_prefix="Validation"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # Increased figure size for better visibility

    # Plot Loss
    axes[0].plot(epochs, losses, marker='o', linestyle='-', color='skyblue')
    axes[0].set_title(f'{title_prefix} Loss over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)

    # Plot Accuracy
    axes[1].plot(epochs, accuracies, marker='o', linestyle='-', color='lightcoral')
    axes[1].set_title(f'{title_prefix} Accuracy over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True)

    # Plot UAR
    axes[2].plot(epochs, uars, marker='o', linestyle='-', color='lightgreen')
    axes[2].set_title(f'{title_prefix} UAR over Epochs')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('UAR')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title_prefix.lower()}_metrics_over_epochs.png'))
    plt.close(fig) # Close the figure to free memory

def plot_confusion_matrix(cm, title, output_path, labels=["Sober (0)", "Drunk (1)"]):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=labels, yticklabels=labels, linewidths=.5) # Added linewidths for better grid visibility
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.savefig(output_path)
    plt.close() # Close the figure to free memory

def plot_confusion_matrix_advanced(cm, title, output_path, labels=["Sober (0)", "Drunk (1)"]):
    """
    Plots a confusion matrix heatmap including raw counts and percentages for
    Sensitivity, Specificity, False Positive Rate, and False Negative Rate.
    Assumes CM is structured as: [[TN, FP], [FN, TP]] for binary classification.
    """
    plt.figure(figsize=(9, 7)) # Slightly larger for more text

    # Extract values from the confusion matrix
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    # Calculate metrics
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0 # True Positive Rate (Recall for positive class)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0 # True Negative Rate (Recall for negative class)
    false_positive_rate = FP / (FP + TN) if (FP + TN) > 0 else 0 # FPR
    false_negative_rate = FN / (FN + TP) if (FN + TP) > 0 else 0 # FNR

    # Create annotations with raw counts and percentages
    annot_labels = np.array([
        [f'{TN}\n(Spec: {specificity:.1%})', f'{FP}\n(FPR: {false_positive_rate:.1%})'],
        [f'{FN}\n(FNR: {false_negative_rate:.1%})', f'{TP}\n(Sens: {sensitivity:.1%})']
    ])

    sns.heatmap(cm, annot=annot_labels, fmt='s', cmap='Blues', cbar=True,
                xticklabels=labels, yticklabels=labels, linewidths=.5,
                linecolor='black', square=True) # Added square=True for better aspect ratio

    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# --- Main execution ---
# Set the directory where your 'file.out' is located
log_directory = r'D:\Uni\Lab\model\v9' # <--- IMPORTANT: SET YOUR DIRECTORY HERE

file_path = os.path.join(log_directory, 'finetune_job.16246516.out') 
plots_output_dir = os.path.join(log_directory, 'plots')

# Ensure the plots output directory exists
os.makedirs(plots_output_dir, exist_ok=True)

if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        outfile_content = f.read()
else:
    print(f"Error: The file '{file_path}' was not found.")
    print("Please ensure 'file.out' is in the specified log_directory, or provide the full path to the directory.")
    exit() 

# Parse the log content
parsed_data = parse_log_content(outfile_content)

# Plot Validation Metrics
if parsed_data["epochs"]:
    plot_metrics_over_epochs(
        parsed_data["epochs"],
        parsed_data["val_losses"],
        parsed_data["val_accuracies"],
        parsed_data["val_uars"],
        output_dir=plots_output_dir,
        title_prefix="Validation"
    )

# Plot Confusion Matrix for Best Validation UAR
if parsed_data["best_val_cm"] is not None:
    plot_confusion_matrix(
        parsed_data["best_val_cm"],
        f'Confusion Matrix - Best Validation UAR (Epoch {parsed_data["best_val_uar_epoch"]})',
        os.path.join(plots_output_dir, 'best_val_confusion_matrix.png')
    )
    # New advanced confusion matrix plot for Best Validation UAR
    plot_confusion_matrix_advanced(
        parsed_data["best_val_cm"],
        f'Advanced Confusion Matrix - Best Validation UAR (Epoch {parsed_data["best_val_uar_epoch"]})',
        os.path.join(plots_output_dir, 'best_val_confusion_matrix_advanced.png')
    )

# Plot Confusion Matrix for Test Set
if parsed_data["test_cm"] is not None:
    plot_confusion_matrix(
        parsed_data["test_cm"],
        f'Confusion Matrix - Test Set (Accuracy: {parsed_data["test_accuracy"]:.4f}, UAR: {parsed_data["test_uar"]:.4f})',
        os.path.join(plots_output_dir, 'test_confusion_matrix.png')
    )
    # New advanced confusion matrix plot for Test Set
    plot_confusion_matrix_advanced(
        parsed_data["test_cm"],
        f'Advanced Confusion Matrix - Test Set (Accuracy: {parsed_data["test_accuracy"]:.4f}, UAR: {parsed_data["test_uar"]:.4f})',
        os.path.join(plots_output_dir, 'test_confusion_matrix_advanced.png')
    )

print("\n--- Visualizations Generated and Saved ---")
print(f"Plots saved to: {plots_output_dir}")

