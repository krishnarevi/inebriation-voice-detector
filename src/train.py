"""
Main script to train and evaluate the inebriation voice detector model.
"""
import os
import time
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from config import (
    DEVICE, TRAIN_DIR, VAL_DIR, TEST_DIR, OUTPUT_DIR, 
    LEARNING_RATE, EPOCHS, set_seed
)
from data_utils import get_data_loaders
from model import create_model, WeightedBinaryCrossEntropyLoss, train_epoch, evaluate

def plot_metrics(train_metrics, val_metrics, metric_name, title, filename):
    """Plot training and validation metrics over epochs"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics, label='Train')
    plt.plot(val_metrics, label='Validation')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.capitalize())
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def plot_confusion_matrix(cm, classes, filename):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, filename):
    """Plot ROC curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def plot_probability_distribution(probs, labels, filename):
    """Plot probability distribution for each class"""
    plt.figure(figsize=(10, 6))
    for i, class_name in [(0, 'SOBER'), (1, 'DRUNK')]:
        class_probs = probs[labels == i]
        if len(class_probs) > 0:
            sns.kdeplot(class_probs, label=class_name, fill=True)
    plt.title('Probability Distribution by Class')
    plt.xlabel('Predicted Probability of DRUNK')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def main():
    """Main function to run training and evaluation"""
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(TRAIN_DIR, VAL_DIR, TEST_DIR)
    
    # Create model
    model = create_model()
    model = model.to(DEVICE)
    
    # Set up loss function and optimizer
    criterion = WeightedBinaryCrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    print("\nOptimizer Configuration:")
    print("  Type: Adam")
    print(f"  Initial learning rate: {LEARNING_RATE}")
    print("  Learning rate scheduler: StepLR (step_size=10, gamma=0.1)")
    
    # Initialize metrics storage
    history = {
        'train_loss': [], 'train_acc': [], 'train_precision': [], 'train_recall': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': []
    }
    
    # Best model tracking
    best_val_f1 = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
    
    # Training loop
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # Train one epoch
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, epoch)
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, criterion)
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{EPOCHS} Results:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"Prec: {train_metrics['precision']:.4f}, Rec: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"Prec: {val_metrics['precision']:.4f}, Rec: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # Save metrics history
        for key, value in train_metrics.items():
            if key in history:
                history[f'train_{key}'].append(value)
        
        for key, value in val_metrics.items():
            if isinstance(value, (int, float)) and key in history:
                history[f'val_{key}'].append(value)
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved with validation F1: {best_val_f1:.4f}")
    
    # Training complete
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(best_model_path))
    
    # Evaluate on test set
    print("\nEvaluating best model on test set...")
    test_metrics = evaluate(model, test_loader, criterion)
    
    print("\nTest Set Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  ROC AUC: {test_metrics['roc']['auc']:.4f}")
    
    # Plot metrics
    plot_metrics(history['train_loss'], history['val_loss'], 'loss', 'Loss Over Epochs', 'loss_plot.png')
    plot_metrics(history['train_acc'], history['val_acc'], 'accuracy', 'Accuracy Over Epochs', 'accuracy_plot.png')
    plot_metrics(history['train_f1'], history['val_f1'], 'f1', 'F1 Score Over Epochs', 'f1_plot.png')
    
    # Plot confusion matrix
    plot_confusion_matrix(test_metrics['confusion_matrix'], ['SOBER', 'DRUNK'], 'confusion_matrix.png')
    
    # Plot ROC curve
    plot_roc_curve(test_metrics['roc']['fpr'], test_metrics['roc']['tpr'], 
                  test_metrics['roc']['auc'], 'roc_curve.png')
    
    # Plot probability distribution
    plot_probability_distribution(test_metrics['probabilities'], test_metrics['true_labels'], 
                                 'probability_distribution.png')
    
    print(f"\nResults and plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()