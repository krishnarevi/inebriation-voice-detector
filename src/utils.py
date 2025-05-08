import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pth.tar'):
    """Save model checkpoint"""
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(output_dir, 'model_best.pth.tar')
        # Copy best model
        torch.save(state, best_filepath)

def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """Load model checkpoint"""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No checkpoint found at '{filepath}'")
    
    checkpoint = torch.load(filepath)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_val_acc = checkpoint.get('best_val_acc', 0.0)
    
    print(f"Loaded checkpoint from epoch {epoch} with validation accuracy {best_val_acc:.4f}")
    
    return epoch, best_val_acc

def plot_training_history(train_losses, val_losses, train_accs, val_accs, output_dir):
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('Training and Validation Loss')
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def evaluate_model(model, dataloader, device, classes):
    """Evaluate model and create visualizations"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probabilities for the positive class
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Generate and print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    
    return {
        'accuracy': np.mean(all_preds == all_labels),
        'confusion_matrix': cm,
        'classification_report': classification_report(all_labels, all_preds, target_names=classes, output_dict=True),
        'roc_auc': roc_auc
    }

def visualize_spectrograms(dataloader, num_samples=10, classes=None):
    """Visualize sample 3-channel spectrograms from the dataset"""
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    
    n = min(num_samples, images.shape[0])
    
    plt.figure(figsize=(15, 3 * (n // 5 + 1)))
    for i in range(n):
        plt.subplot(n // 5 + 1, 5, i + 1)
        
        img = images[i].permute(1, 2, 0).numpy()  # Convert from [C, H, W] to [H, W, C]
        plt.imshow(img)
        if classes:
            plt.title(f"Class: {classes[labels[i]]}")
        else:
            plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def calculate_class_weights(dataset):
    """Calculate weights for imbalanced dataset"""
    labels = []
    for _, label in dataset:
        labels.append(label)
    
    labels = np.array(labels)
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    
    # Compute weights inversely proportional to class frequencies
    weights = total_samples / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights)

if __name__ == "__main__":
    # Test the plot function
    train_losses = [0.9, 0.8, 0.7, 0.6, 0.5]
    val_losses = [0.95, 0.85, 0.75, 0.65, 0.55]
    train_accs = [0.6, 0.7, 0.8, 0.85, 0.9]
    val_accs = [0.55, 0.65, 0.75, 0.8, 0.85]
    
    os.makedirs('./test_output', exist_ok=True)
    plot_training_history(train_losses, val_losses, train_accs, val_accs, './test_output')
    print("Test plot saved to ./test_output/training_history.png")