"""
Dataset classes and data loading utilities for the inebriation voice detector.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from PIL import Image
from config import CLASS_TO_IDX, IDX_TO_CLASS, DEVICE, BATCH_SIZE, OUTPUT_DIR

# Define image transformations
def get_transforms():
    """Get image transformations for training and validation"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    # For visualization purposes (optional)
    vis_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    return transform, vis_transform

class SpectrogramDataset(Dataset):
    """Dataset class for loading spectrogram images"""
    def __init__(self, root_dir, transform=None, vis_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.vis_transform = vis_transform  # For visualization
        self.samples = []
        
        # Load all file paths and labels
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                class_idx = CLASS_TO_IDX[class_name]
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

def calculate_sampling_weights(dataset):
    """Calculate sampling weights for balanced sampling"""
    # Count samples in each class
    class_counts = {i: 0 for i in range(len(CLASS_TO_IDX))}
    for _, label in dataset:
        class_counts[label] += 1
    
    print("\nClass distribution before balancing:")
    for i, count in class_counts.items():
        print(f"  {IDX_TO_CLASS[i]}: {count} samples")
    
    # Calculate class weights (inverse frequency)
    num_samples = len(dataset)
    class_weights = {i: num_samples / count for i, count in class_counts.items()}
    print("\nClass weights (inverse frequency):")
    for i, weight in class_weights.items():
        print(f"  {IDX_TO_CLASS[i]}: {weight:.4f}")
    
    # Assign weight to each sample
    sample_weights = [class_weights[label] for _, label in dataset]
    
    # Visualize weights
    df = pd.DataFrame({
        'Class': [IDX_TO_CLASS[label] for _, label in dataset],
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

def get_data_loaders(train_dir, val_dir, test_dir):
    """Create and return data loaders for training, validation, and testing"""
    # Get transforms
    transform, vis_transform = get_transforms()
    
    # Create datasets
    train_dataset = SpectrogramDataset(train_dir, transform=transform)
    val_dataset = SpectrogramDataset(val_dir, transform=transform)
    test_dataset = SpectrogramDataset(test_dir, transform=transform)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Calculate sampling weights
    sample_weights, original_counts, class_weights = calculate_sampling_weights(train_dataset)
    
    # Create weighted sampler for balanced training
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    # Create data loaders
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
    
    return train_loader, val_loader, test_loader