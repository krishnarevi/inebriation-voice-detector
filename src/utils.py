import torch
import random
import numpy as np
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from config import *

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def get_weighted_sampler(dataset):
    # Count class frequencies
    label_counts = Counter([label for _, label in dataset.samples])
    
    # Avoid division by zero
    total_classes = len(dataset.classes)
    class_counts = torch.tensor([label_counts.get(i, 0) for i in range(total_classes)], dtype=torch.float)
    
    # Compute weights
    class_weights = 1.0 / (class_counts + 1e-6)  # Add small value to avoid division by zero
    class_weights = class_weights / class_weights.sum()
    
    sample_weights = torch.tensor([class_weights[label] for _, label in dataset.samples])
    
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)
    return sampler


def calculate_metrics(labels, preds):
    """
    Calculates Precision, Recall, F1 Score, and Unweighted Average Recall (UAR) from confusion matrix.

    Args:
        labels (array): Ground truth labels.
        preds (array): Predicted labels.

    Returns:
        dict: Dictionary containing the metrics.
    """
    cm = confusion_matrix(labels, preds)
    
    if cm.shape == (2, 2):  # binary classification
        tn, fp, fn, tp = cm.ravel()
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        uar = (tpr + tnr) / 2
    else:  # Handle case when one class is missing in predictions
        uar = accuracy_score(labels, preds)  # fallback

    # Calculate Precision, Recall, F1
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    # Return all the metrics as a dictionary
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'uar': uar
    }


def plot_results(metrics, class_names):
    plt.figure(figsize=(8, 4))
    
    # Confusion matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', 
                cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    # Metrics summary
    plt.subplot(1, 2, 2)
    plt.axis('off')
    metrics_text = (
        "Test Set Results:\n\n"
        f"Accuracy: {metrics['accuracy']:.4f}\n"
        f"UAR: {metrics['uar']:.4f}\n"
        f"Loss: {metrics['loss']:.4f}\n"
    )
    plt.text(0.1, 0.5, metrics_text, fontsize=12, va='center')
    
    plt.tight_layout()
    plt.savefig("output/test_results.png")
    plt.show()


if __name__ == "__main__":
    print("Testing utils:")
    set_seed(42)