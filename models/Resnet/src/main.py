import os
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data_exploration import explore_dataset, visualize_spectrograms
from dataset import SpectrogramDataset
from model import resnet18_classifier
from utils import *
from config import *  
from train import train_model
from evaluate import evaluate


def main():
    set_seed(42)
    # 1. Data Exploration
    print("=== Exploring Dataset ===")
    train_files, train_labels, train_counts = explore_dataset(TRAIN_DIR, "Training")
    val_files, val_labels, val_counts = explore_dataset(VAL_DIR, "Validation")
    test_files, test_labels, test_counts = explore_dataset(TEST_DIR, "Test")
    
    # Visualize sample spectrograms
    visualize_spectrograms(
        train_files, 
        train_labels, 
        num_samples=5, 
        save_path=os.path.join(OUTPUT_DIR, "sample_spectrograms.png")
    )

    # 2. Prepare Data Loaders
    print("\n=== Preparing Data Loaders ===")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    # Initialize datasets
    train_dataset = SpectrogramDataset(TRAIN_DIR, transform)
    val_dataset = SpectrogramDataset(VAL_DIR, transform)
    test_dataset = SpectrogramDataset(TEST_DIR, transform)

    # Create balanced sampler
    sampler = get_weighted_sampler(train_dataset)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training samples: {len(train_dataset)} ({len(train_loader)} batches)")
    print(f"Validation samples: {len(val_dataset)} ({len(val_loader)} batches)")
    print(f"Test samples: {len(test_dataset)} ({len(test_loader)} batches)")

    # 3. Initialize Model and Training Setup
    print("\n=== Initializing Model ===")
    model = resnet18_classifier()
    model.to(device)
    
    # Handle class imbalance
    pos_weight = torch.tensor([train_counts['SOBER'] / train_counts['DRUNK']]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer for fine-tuned layers only
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=INITIAL_LR,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=PATIENCE,
        factor=0.5
    )

    print("\nTraining setup:")
    print(f"Model: {model.__class__.__name__}")
    print(f"Loss: BCEWithLogitsLoss (pos_weight={pos_weight.item():.2f})")
    print(f"Optimizer: Adam(lr={INITIAL_LR}, weight_decay=1e-4)")
    print("LR scheduler: ReduceLROnPlateau (on val loss)")
    print("\n=== Initializing Training ===")
    print(f"Device: {device}")
    print(f"Classes: {classes}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Initial LR: {INITIAL_LR}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Patience: {PATIENCE} epochs\n")

    # 4. Train the Model
    print(f"\n=== Training for {NUM_EPOCHS} epochs ===")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,

    )

    # 5. Evaluate on Test Set
    print("\n=== Final Evaluation ===")
    # Load best model weights
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth")))
    
    test_metrics = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,

    )

    # Print and save results
    print("\nTest Results:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"UAR: {test_metrics['uar']:.4f}")
    print("Confusion Matrix:")
    print(test_metrics['confusion_matrix'])

    # Save metrics
    torch.save(test_metrics, os.path.join(OUTPUT_DIR, "test_metrics.pth"))

    # 6. Visualization of results
    plot_results(
        test_metrics,
        class_names=classes
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")
        raise