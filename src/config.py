"""
Configuration settings for the inebriation voice detector project.
"""
import os
import torch
import random
import numpy as np

# Classes and mappings
CLASSES = ['SOBER', 'DRUNK']  # SOBER = 0, DRUNK = 1
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}

# Data paths - Change these to your actual data paths
DATA_ROOT = "/Users/krishna/University/Sem2/Phonetics_Lab/Code/inebriation-voice-detector/data/processed"
TRAIN_DIR = os.path.join(DATA_ROOT, "TRAIN")
VAL_DIR = os.path.join(DATA_ROOT, "VALIDATION")
TEST_DIR = os.path.join(DATA_ROOT, "TEST")

# Output directory
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training parameters
BATCH_SIZE = 100
LEARNING_RATE = 0.001
EPOCHS = 2
POSITIVE_WEIGHT = 0.9  # Weight for DRUNK class
NEGATIVE_WEIGHT = 0.1  # Weight for SOBER class

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")