import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class SpectrogramDataset(Dataset):
    """Dataset for loading spectrograms of speech signals."""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the spectrograms.
                Structure: root_dir/SOBER/, root_dir/DRUNK/
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['SOBER', 'DRUNK']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((
                        os.path.join(class_dir, img_name),
                        self.class_to_idx[class_name]
                    ))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load spectrogram as grayscale image
        image = Image.open(img_path).convert('RGB')  # 'L' mode is for grayscale
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms():
    """Returns transformations for training and validation/test sets."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return train_transform, val_transform


def create_dataloaders(data_dir, batch_size=100, num_workers=4):
    """
    Creates and returns dataloaders for training, validation, and test sets.
    
    Args:
        data_dir: Root directory containing train, val, test folders
        batch_size: Batch size for the dataloaders
        num_workers: Number of worker threads for loading data
        
    Returns:
        train_loader, val_loader, test_loader
    """
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = SpectrogramDataset(
        os.path.join(data_dir, 'TRAIN'),
        transform=train_transform
    )
    
    val_dataset = SpectrogramDataset(
        os.path.join(data_dir, 'VALIDATION'),
        transform=val_transform
    )
    
    test_dataset = SpectrogramDataset(
        os.path.join(data_dir, 'TEST'),
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test the dataset and dataloader
    import matplotlib.pyplot as plt
    
    data_dir = "./data/processed"  # Adjust path as needed
    train_transform, _ = get_transforms()
    
    # Create a small test dataset
    dataset = SpectrogramDataset(
        os.path.join(data_dir, "TRAIN"),
        transform=train_transform
    )
    
    # Print dataset info
    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.classes}")
    
    # Get a sample
    if len(dataset) > 0:
        image, label = dataset[0]
        print(f"Sample shape: {image.shape}, Label: {label} ({dataset.classes[label]})")
        
        # Visualize the sample
        plt.figure(figsize=(6, 6))
        # Convert tensor to numpy and adjust for display
        img_np = np.transpose(image.numpy(), (1, 2, 0))  # convert CHW to HWC
        plt.imshow((img_np * 0.5 + 0.5))  # denormalize for display

        plt.title(f"Class: {dataset.classes[label]}")
        plt.colorbar(label='Amplitude')
        plt.tight_layout()
        plt.show()
    
    # Test the dataloader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for images, labels in loader:
        print(f"Batch shapes: {images.shape}, {labels.shape}")
        break