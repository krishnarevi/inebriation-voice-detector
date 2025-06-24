from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import os
from PIL import Image
from config import *
from utils import get_weighted_sampler


class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['SOBER', 'DRUNK']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._make_dataset()
        
    def _make_dataset(self):
        samples = []
        for class_name in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path) and class_name in self.class_to_idx:
                class_idx = self.class_to_idx[class_name]
                for filename in os.listdir(class_path):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        samples.append((
                            os.path.join(class_path, filename),
                            class_idx
                        ))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def create_data_loaders():
    transform = get_transform()
    
    train_dataset = SpectrogramDataset(TRAIN_DIR, transform)
    val_dataset = SpectrogramDataset(VAL_DIR, transform)
    test_dataset = SpectrogramDataset(TEST_DIR, transform)

    sampler = get_weighted_sampler(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Testing dataset:")
    dataset = SpectrogramDataset(TRAIN_DIR, transform=get_transform())
    print(f"Dataset length: {len(dataset)}")
    sample, label = dataset[0]
    print(f"Sample shape: {sample.shape}, Label: {label}")

    train_loader, val_loader, test_loader = create_data_loaders()
    batch = next(iter(train_loader))
    print(f"Batch shapes - images: {batch[0].shape}, labels: {batch[1].shape}")