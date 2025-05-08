import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import librosa
import glob
from models.ast_models import ASTModel  # Make sure AST model is installed and imported

class SpectrogramDataset(Dataset):
    def __init__(self, data_dir, sr=16000, t_dim=1200):
        self.files = glob.glob(os.path.join(data_dir, '*.wav'))
        self.labels = [1 if 'drunk' in f else 0 for f in self.files]  # crude label parse
        self.sr = sr
        self.t_dim = t_dim

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path = self.files[idx]
        y, sr = librosa.load(audio_path, sr=self.sr)
        mel = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
        logmel = librosa.power_to_db(mel)

        # Pad/crop to t_dim
        if logmel.shape[1] < self.t_dim:
            pad = self.t_dim - logmel.shape[1]
            logmel = np.pad(logmel, ((0, 0), (0, pad)), mode='constant')
        else:
            logmel = logmel[:, :self.t_dim]

        logmel = torch.tensor(logmel).unsqueeze(0).float()  # [1, 128, t_dim]
        label = torch.tensor(self.labels[idx]).long()
        return logmel, label


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss, correct = 0.0, 0
    total = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return running_loss / total, correct / total


def main():
    data_dir = './data/audio'  # path to your wav files
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = SpectrogramDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    model = ASTModel(
        label_dim=2,
        input_tdim=1200,
        imagenet_pretrain=False,
        audioset_pretrain=True
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        loss, acc = train_one_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={acc*100:.2f}%")

    torch.save(model.state_dict(), 'finetuned_ast.pth')


if __name__ == "__main__":
    main()
