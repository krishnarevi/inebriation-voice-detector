import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

# ----- Dataset Class -----
class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.classes = ['SOBER', 'DRUNK']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# ----- Transforms -----
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

# ----- Training Function -----
def train(model, loader, criterion, optimizer, device):
    model.train()
    total, correct, loss_total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        loss_total += loss.item() * y.size(0)

    return loss_total / total, correct / total

# ----- Validation Function -----
def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, loss_total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)

            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            loss_total += loss.item() * y.size(0)

    return loss_total / total, correct / total

# ----- Main -----
def main():
    data_root = "./data/processed"
    batch_size = 100
    num_epochs = 30
    lr = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    train_ds = SpectrogramDataset(os.path.join(data_root, "TRAIN"), transform=get_transforms())
    val_ds = SpectrogramDataset(os.path.join(data_root, "VALIDATION"), transform=get_transforms())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

    # Model
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # Loss and optimizer
    class_weights = torch.tensor([1.0, 1.0]).to(device)  # Optional: adjust if class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1:02d}: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}, Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

if __name__ == "__main__":
    main()
