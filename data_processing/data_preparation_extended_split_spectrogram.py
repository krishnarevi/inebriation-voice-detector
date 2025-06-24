import os
import shutil
import random
import numpy as np
import torch

# ---------- SETTINGS ----------
DATA_ROOT = r"D:\Uni\Lab\inebriation-voice-detector\data\processed"
TRAIN_DIR = os.path.join(DATA_ROOT,  'TRAIN')
VAL_DIR = os.path.join(DATA_ROOT,  'VALIDATION')
TEST_DIR = os.path.join(DATA_ROOT, 'TEST')

OUTPUT_ROOT = r"D:\Uni\Lab\inebriation-voice-detector\output"
EX_VAL_DIR = os.path.join(OUTPUT_ROOT, "ex_val")

classes = ['SOBER', 'DRUNK']
NUM_VAL_SAMPLES = 3000  # total to sample from validation and move to train

# ---------- SET SEED ----------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Seed set to {seed}")

set_seed(42)

# ---------- CREATE DIR STRUCTURE ----------
def create_subdirs(base_dir, classes):
    for cls in classes:
        os.makedirs(os.path.join(base_dir, cls), exist_ok=True)

train_out = os.path.join(EX_VAL_DIR, 'TRAIN')
val_out = os.path.join(EX_VAL_DIR, 'VAL')
test_out = os.path.join(EX_VAL_DIR, 'TEST')

for path in [train_out, val_out, test_out]:
    create_subdirs(path, classes)

# ---------- COPY ORIGINAL TRAIN ----------
print("[INFO] Copying original TRAIN files...")
for cls in classes:
    src = os.path.join(TRAIN_DIR, cls)
    dst = os.path.join(train_out, cls)
    for fname in os.listdir(src):
        fsrc = os.path.join(src, fname)
        fdst = os.path.join(dst, fname)
        if os.path.isfile(fsrc):
            shutil.copy2(fsrc, fdst)

# ---------- BALANCED SAMPLE FROM VAL ----------
print("[INFO] Sampling from VAL and redistributing...")

val_file_map = {}
total_val_files = 0

# Gather all val files
for cls in classes:
    cls_dir = os.path.join(VAL_DIR, cls)
    files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]
    val_file_map[cls] = files
    total_val_files += len(files)

# Calculate proportional sample counts
val_sample_counts = {}
for cls in classes:
    val_sample_counts[cls] = int((len(val_file_map[cls]) / total_val_files) * NUM_VAL_SAMPLES)

# Fix rounding issue (ensure total is exactly NUM_VAL_SAMPLES)
total_selected = sum(val_sample_counts.values())
if total_selected < NUM_VAL_SAMPLES:
    diff = NUM_VAL_SAMPLES - total_selected
    largest_class = max(val_sample_counts, key=lambda k: len(val_file_map[k]))
    val_sample_counts[largest_class] += diff

# Shuffle, split, and copy files
for cls in classes:
    files = val_file_map[cls]
    random.shuffle(files)
    selected = files[:val_sample_counts[cls]]
    remaining = files[val_sample_counts[cls]:]

    # Selected go to TRAIN
    for fpath in selected:
        fname = os.path.basename(fpath)
        shutil.copy2(fpath, os.path.join(train_out, cls, fname))

    # Remaining go to VAL
    for fpath in remaining:
        fname = os.path.basename(fpath)
        shutil.copy2(fpath, os.path.join(val_out, cls, fname))

    print(f"[INFO] {cls}: moved {len(selected)} to TRAIN, kept {len(remaining)} in VAL")

# ---------- COPY TEST ----------
print("[INFO] Copying TEST files...")
for cls in classes:
    src = os.path.join(TEST_DIR, cls)
    dst = os.path.join(test_out, cls)
    for fname in os.listdir(src):
        fsrc = os.path.join(src, fname)
        fdst = os.path.join(dst, fname)
        if os.path.isfile(fsrc):
            shutil.copy2(fsrc, fdst)

print(f"[DONE] New dataset created at: {EX_VAL_DIR}")
