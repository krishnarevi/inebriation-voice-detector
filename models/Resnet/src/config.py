import os
import torch
# Constants
DATA_ROOT = r"D:\Uni\Lab\inebriation-voice-detector\data\processed"
TRAIN_DIR = os.path.join(DATA_ROOT, "TRAIN")
VAL_DIR = os.path.join(DATA_ROOT, "VALIDATION")
TEST_DIR = os.path.join(DATA_ROOT, "TEST")
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class labels
classes = ['SOBER', 'DRUNK']
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
idx_to_class = {idx: cls for idx, cls in enumerate(classes)}

# Training params
BATCH_SIZE = 100
INITIAL_LR = 5e-4 #3e-4 # .0001
PATIENCE = 5
NUM_EPOCHS = 30

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    print("Configuration loaded successfully")
    print(f"DATA_ROOT: {DATA_ROOT}")