import os
# Set HF_HOME environment variable to a local path (e.g., /tmp) to avoid file locking issues
# This must be set *before* importing transformers
os.environ["HF_HOME"] = "/tmp" 

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, get_linear_schedule_with_warmup
import librosa
import soundfile as sf
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
from tqdm import tqdm
import warnings
from datetime import datetime
from audiomentations import Compose, AddGaussianNoise, PitchShift, Gain

# Suppress specific librosa warning about audioread backend
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
# Suppress the transformers warning about uninitialized weights in the classification head
warnings.filterwarnings(
    "ignore", 
    message="Some weights of the model checkpoint at.*were not initialized.*", 
    category=UserWarning, 
    module="transformers"
)

# --- Configuration ---
RAW_AUDIO_ROOT = "/fast/krevi/ALC_extended_split"
BASE_OUTPUT_DIR = "/fast/krevi/v5_ext_split" 

MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-german" 
NUM_LABELS = 2 # Sober (0), Drunk (1)
SAMPLING_RATE = 16000 # Wav2Vec2 models are typically trained on 16kHz audio

# Training parameters
BATCH_SIZE = 4 
GRADIENT_ACCUMULATION_STEPS = 2 
NUM_EPOCHS = 20
LEARNING_RATE = 5e-5

# --- Global Initialization ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}") 

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

# Initialize Automatic Mixed Precision Scaler (Corrected for FutureWarning)
scaler = torch.amp.GradScaler(device='cuda') # <--- CORRECTED: Updated GradScaler initialization


# --- Custom Dataset (Directly reads from folders) ---
class AudioDatasetFromFolders(Dataset):
    def __init__(self, audio_root, split_name, processor):
        self.processor = processor
        self.data = []

        label_folders = {"SOBER": 0, "DRUNK": 1}

        split_path = os.path.join(audio_root, split_name)
        if not os.path.isdir(split_path):
            raise ValueError(f"Split folder '{split_path}' not found. Please check RAW_AUDIO_ROOT and split_name.")

        print(f"Scanning audio files in {split_path}...")
        for folder_name, label_id in label_folders.items():
            label_path = os.path.join(split_path, folder_name)
            if not os.path.isdir(label_path):
                print(f"Warning: Label folder '{label_path}' not found. Skipping files from this label in {split_name}.")
                continue

            for filename in os.listdir(label_path):
                if filename.lower().endswith(".wav"):
                    full_audio_path = os.path.join(label_path, filename)
                    if os.path.exists(full_audio_path):
                        self.data.append((full_audio_path, label_id))
                    else:
                        print(f"Warning: Audio file not found: {full_audio_path}. Skipping.")
        
        if len(self.data) == 0:
            raise ValueError(f"No audio files found for '{split_name}' split in '{audio_root}'. Check folder structure and WAV files.")
        else:
            print(f"Found {len(self.data)} audio files for {split_name} split.")

        # --- Initialize augmentation only for the TRAIN split ---
        self.augment = None
        if split_name == "TRAIN":
            self.augment = Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
                Gain(min_gain_db=-6.0, max_gain_db=6.0, p=0.5), # <--- CORRECTED: Parameter names for Gain
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, label = self.data[idx]
        speech, sr = librosa.load(audio_path, sr=SAMPLING_RATE)

        # --- Apply augmentation if it's the training set ---
        if self.augment:
            # audiomentations expects samples to be float32
            speech = self.augment(samples=speech, sample_rate=sr)
            
        return {"input_values": speech, "labels": label}

# --- Custom Collate Function for DataLoader ---
def collate_fn(batch):
    input_features = [item["input_values"] for item in batch]
    labels = [item["labels"] for item in batch]

    batch = processor(input_features, sampling_rate=SAMPLING_RATE, padding=True, return_tensors="pt")
    batch["labels"] = torch.tensor(labels, dtype=torch.long)
    return batch

# --- Main Execution Block ---
if __name__ == '__main__':
    # Define a unique output directory for this run using a timestamp
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"run_{current_time}")

    # Ensure the specific OUTPUT_DIR for this run exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Results for this run will be saved in: {OUTPUT_DIR}")
    
    # --- Dataset and DataLoader Creation ---
    train_dataset = AudioDatasetFromFolders(RAW_AUDIO_ROOT, "TRAIN", processor)
    val_dataset = AudioDatasetFromFolders(RAW_AUDIO_ROOT, "VALIDATION", processor) 
    test_dataset = AudioDatasetFromFolders(RAW_AUDIO_ROOT, "TEST", processor)

    num_workers = os.cpu_count() // 2 
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # --- Initialize model ---
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model.to(DEVICE)

    # --- Optimizer and Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = len(train_dataloader) * NUM_EPOCHS 
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * 0.1), # 10% warmup
        num_training_steps=num_training_steps,
    )

    # --- Training Loop ---
    best_val_uar = -1.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training")

        model.zero_grad() 

        for step, batch in enumerate(progress_bar):
            input_values = batch["input_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            with torch.cuda.amp.autocast():
                outputs = model(input_values, labels=labels)
                loss = outputs.loss

            loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS 

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(train_dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
                
                scaler.step(optimizer)
                scaler.update() 
                lr_scheduler.step()
                model.zero_grad()
            
            progress_bar.set_postfix({"loss": total_loss / (step + 1)})

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}: Average Training Loss: {avg_train_loss:.4f}")

        # --- Validation Loop ---
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0
        progress_bar_val = tqdm(val_dataloader, desc=f"Epoch {epoch+1} Validation")

        for batch in progress_bar_val:
            input_values = batch["input_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs = model(input_values, labels=labels)
                
            val_loss += outputs.loss.item()
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            val_preds.extend(predictions.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_dataloader)
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_uar = recall_score(val_labels, val_preds, average='macro') 
        val_cm = confusion_matrix(val_labels, val_preds)

        print(f"Epoch {epoch+1}: Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}, UAR: {val_uar:.4f}")
        print(f"Validation Confusion Matrix:\n{val_cm}")

        # Save best model based on validation UAR
        if val_uar > best_val_uar:
            best_val_uar = val_uar
            os.makedirs(os.path.join(OUTPUT_DIR, "best_wav2vec2_model"), exist_ok=True) 
            model.save_pretrained(os.path.join(OUTPUT_DIR, "best_wav2vec2_model"))
            processor.save_pretrained(os.path.join(OUTPUT_DIR, "best_wav2vec2_model"))
            print(f"New best validation UAR: {best_val_uar:.4f}. Model saved to {OUTPUT_DIR}/best_wav2vec2_model!")

    # --- Testing Loop (after training) ---
    print("\n--- Running Test Set Evaluation ---")
    
    try:
        model = Wav2Vec2ForSequenceClassification.from_pretrained(os.path.join(OUTPUT_DIR, "best_wav2vec2_model"), num_labels=NUM_LABELS)
        processor = Wav2Vec2Processor.from_pretrained(os.path.join(OUTPUT_DIR, "best_wav2vec2_model"))
        model.to(DEVICE)
        print("Loaded best saved model for final test evaluation.")
    except Exception as e:
        print(f"Could not load best saved model for test evaluation: {e}. Ensure a model was saved correctly. Using model from last epoch (or last best if training completed).")

    model.eval()
    test_preds = []
    test_labels = []
    progress_bar_test = tqdm(test_dataloader, desc="Test Evaluation")

    for batch in progress_bar_test:
        input_values = batch["input_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = model(input_values, labels=labels)
            
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        test_preds.extend(predictions.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

    test_accuracy = accuracy_score(test_labels, test_preds)
    test_uar = recall_score(test_labels, test_preds, average='macro')
    test_cm = confusion_matrix(test_labels, test_preds)

    print("\nTest Results:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"UAR: {test_uar:.4f}")
    print(f"Confusion Matrix:\n{test_cm}")

    print("\n--- Training and Evaluation Complete ---")
    print(f"You can find the best performing model (based on validation UAR) saved in '{OUTPUT_DIR}/best_wav2vec2_model' directory.")