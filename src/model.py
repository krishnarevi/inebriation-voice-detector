"""
Model architecture, loss functions, and evaluation metrics for the inebriation voice detector.
"""
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from config import DEVICE, POSITIVE_WEIGHT, NEGATIVE_WEIGHT

class WeightedBinaryCrossEntropyLoss(nn.Module):
    """Weighted binary cross entropy loss for handling class imbalance"""
    def __init__(self, weight_pos=POSITIVE_WEIGHT, weight_neg=NEGATIVE_WEIGHT):
        super(WeightedBinaryCrossEntropyLoss, self).__init__()
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg
        print("\nWeighted Binary Cross Entropy Loss:")
        print(f"  Positive class (DRUNK) weight: {weight_pos}")
        print(f"  Negative class (SOBER) weight: {weight_neg}")
        
    def forward(self, pred, target):
        target = target.float()
        loss = self.weight_pos * target * torch.log(pred + 1e-7) + \
               self.weight_neg * (1 - target) * torch.log(1 - pred + 1e-7)
        return -torch.mean(loss)

def create_model():
    """Create and configure ResNet-18 model for binary classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=True)
    
    print("\nModel Architecture (before modification):")
    print(model)
    
    # Print number of parameters before modification
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Freeze all layers except final ones
    for name, param in model.named_parameters():
        if "fc" not in name:  # Freeze all layers except the fully connected layer
            param.requires_grad = False
    
    # Modify the final fully connected layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    
    print("\nModel Architecture (after modification):")
    print(model)
    
    # Print number of parameters after modification
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model

def train_epoch(model, train_loader, criterion, optimizer, epoch):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    # Use tqdm for progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for inputs, labels in progress_bar:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        outputs = outputs.squeeze()
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item() * inputs.size(0)
        
        # Convert probabilities to binary predictions
        preds = (outputs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Store for metrics calculation
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}", 
            'acc': f"{100 * correct / total:.2f}%"
        })
    
    # Calculate epoch metrics
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100 * correct / total
    
    # Additional metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = {
        'loss': epoch_loss,
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
    }
    
    return metrics

def evaluate(model, data_loader, criterion):
    """Evaluate the model on the given data loader"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # Store raw probabilities for ROC curve
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            # Store raw probabilities
            all_probs.extend(outputs.cpu().numpy())
            
            # Convert probabilities to binary predictions
            preds = (outputs >= 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'loss': running_loss / len(data_loader.dataset),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'confusion_matrix': cm,
        'roc': {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc},
        'probabilities': all_probs,
        'true_labels': all_labels
    }
    
    return metrics