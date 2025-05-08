# %%

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import random
from pathlib import Path
import cv2

# %%
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

set_seed(42)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths to data directories
DATA_ROOT = "/Users/krishna/University/Sem2/Phonetics_Lab/Code/inebriation-voice-detector/data/processed"  # Change this to your actual data path
TRAIN_DIR = os.path.join(DATA_ROOT, "TRAIN")
VAL_DIR = os.path.join(DATA_ROOT, "VALIDATION")
TEST_DIR = os.path.join(DATA_ROOT, "TEST")

# Create output directory for results
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define classes and class indices
classes = ['SOBER', 'DRUNK']  # SOBER = 0, DRUNK = 1
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
idx_to_class = {idx: cls for idx, cls in enumerate(classes)}

print(f"Class mapping: {class_to_idx}")

# %%
# ====================== DATA TRANSFORMS ======================

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])


# %%
# ====================== DATASET CLASS ======================

class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None, vis_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.vis_transform = vis_transform  # For visualization
        self.samples = []
        
        # Load all file paths and labels
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                class_idx = class_to_idx[class_name]
                for filename in os.listdir(class_path):
                    if filename.endswith('.jpg') or filename.endswith('.png'):
                        self.samples.append((os.path.join(class_path, filename), class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        transformed_img = None
        if self.transform:
            transformed_img = self.transform(image)
        
        # For visualization purposes
        vis_img = None
        if self.vis_transform:
            vis_img = self.vis_transform(image)
            return transformed_img, label, vis_img, img_path
        
        return transformed_img, label

# Create datasets
train_dataset = SpectrogramDataset(TRAIN_DIR, transform=transform)
val_dataset = SpectrogramDataset(VAL_DIR, transform=transform)
test_dataset = SpectrogramDataset(TEST_DIR, transform=transform)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# %%
# ====================== BALANCED SAMPLING ======================

def calculate_sampling_weights(dataset):
    """Calculate sampling weights for balanced sampling"""
    # Count samples in each class
    class_counts = {i: 0 for i in range(len(classes))}
    for _, label in dataset:
        class_counts[label] += 1
    
    print("\nClass distribution before balancing:")
    for i, count in class_counts.items():
        print(f"  {idx_to_class[i]}: {count} samples")
    
    # Calculate class weights (inverse frequency)
    num_samples = len(dataset)
    class_weights = {i: num_samples / count for i, count in class_counts.items()}
    print("\nClass weights (inverse frequency):")
    for i, weight in class_weights.items():
        print(f"  {idx_to_class[i]}: {weight:.4f}")
    
    # Assign weight to each sample
    sample_weights = [class_weights[label] for _, label in dataset]
    
    # Visualize weights
    df = pd.DataFrame({
        'Class': [idx_to_class[label] for _, label in dataset],
        'Weight': sample_weights
    })
    
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Class', y='Weight', data=df)
    plt.title("Sampling Weights Distribution")
    plt.ylabel("Weight")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sampling_weights.png"))
    plt.close()
    
    return sample_weights, class_counts, class_weights

# Calculate sampling weights
sample_weights, original_counts, class_weights = calculate_sampling_weights(train_dataset)

# Create weighted sampler for balanced training
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_dataset),
    replacement=True
)



# %%
# ====================== DATA LOADERS ======================

BATCH_SIZE = 100

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=4,
    pin_memory=True if torch.cuda.is_available() else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True if torch.cuda.is_available() else False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True if torch.cuda.is_available() else False
)




# %%
# ====================== MODEL ARCHITECTURE ======================

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


# Create model
model = create_model()
model = model.to(device)


# %%
# ====================== LOSS FUNCTION ======================

class WeightedBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, weight_pos=0.9, weight_neg=0.1):
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


# Set up loss function
criterion = WeightedBinaryCrossEntropyLoss(weight_pos=0.9, weight_neg=0.1)


# %%
# ====================== OPTIMIZER & SCHEDULER ======================

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

print("\nOptimizer Configuration:")
print("  Type: Adam")
print("  Initial learning rate: 0.001")
print("  Learning rate scheduler: StepLR (step_size=10, gamma=0.1)")


# %%
# ====================== TRAINING FUNCTIONS ======================

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
        inputs = inputs.to(device)
        labels = labels.to(device)
        
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
            inputs = inputs.to(device)
            labels = labels.to(device)
            
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

# %%


# %%



