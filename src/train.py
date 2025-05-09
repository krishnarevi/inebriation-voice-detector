import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataloader import create_dataloaders
from model import get_model
from utils import AverageMeter, save_checkpoint, plot_training_history

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    
    losses = AverageMeter('Loss')
    accuracies = AverageMeter('Acc')
    
    pbar = tqdm(dataloader, desc='Training')
    for inputs, targets in pbar:
        inputs = inputs.to(device)  # [batch_size, 3, 224, 224] expected
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == targets).item() / targets.size(0)
        
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        pbar.set_postfix({'loss': losses.avg, 'acc': accuracies.avg})
    
    return losses.avg, accuracies.avg

def validate(model, dataloader, criterion, device):
    """Validate the model on the validation set."""
    model.eval()
    
    losses = AverageMeter('Loss')
    accuracies = AverageMeter('Acc')
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            _, preds = torch.max(outputs, 1)
            acc = torch.sum(preds == targets).item() / targets.size(0)
            
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))
            pbar.set_postfix({'loss': losses.avg, 'acc': accuracies.avg})
    
    return losses.avg, accuracies.avg

def train(args):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # rgb=True  # <-- Ensure dataset returns 3-channel spectrograms
    )
    
    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    
    model = get_model(args.model, num_classes=2, pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    best_val_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step(val_loss)
        is_best = val_acc > best_val_acc
        best_val_acc = max(val_acc, best_val_acc)
        
        if is_best or (epoch + 1) % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_losses': val_losses,
                'val_accs': val_accs,
            }, is_best, args.output_dir)
    
    plot_training_history(train_losses, val_losses, train_accs, val_accs, args.output_dir)
    
    print("\nEvaluating on test set...")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RGB spectrogram classifier')
    
    # Dataset
    parser.add_argument('--data_dir', type=str, default='./data/processed', help='Data directory')
    
    # Model
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet', 'efficientnet'], help='Model type')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights (only for resnet/efficientnet)')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Epoch count')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer choice')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--no_cuda', action='store_true', help='Force CPU mode')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./output', help='Checkpoint/output directory')
    parser.add_argument('--save_freq', type=int, default=5, help='Checkpoint save frequency')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    model = train(args)

