import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, get_linear_schedule_with_warmup
import librosa
import soundfile as sf
import os
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
from tqdm import tqdm
import warnings

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
# Root directory where your TRAIN, TEST, VALIDATION folders are located
RAW_AUDIO_ROOT = r"D:\Uni\Lab\inebriation-voice-detector\data\raw_data"

# Changed MODEL_NAME to a German-specific pre-trained Wav2Vec2 model
MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-german" 
NUM_LABELS = 2 # Sober (0), Drunk (1)
SAMPLING_RATE = 16000 # Wav2Vec2 models are typically trained on 16kHz audio

# Training parameters - ADJUSTED FOR MEMORY
BATCH_SIZE = 2 # Significantly reduced batch size
GRADIENT_ACCUMULATION_STEPS = 4 # Accumulate gradients over 4 batches, simulating an effective batch size of 2*4=8
NUM_EPOCHS = 10 # Start with 10-20, monitor validation performance
LEARNING_RATE = 5e-5

# --- Global Initialization (Moved outside if __name__ == '__main__': for multiprocessing compatibility) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {DEVICE}") # Removed this print statement

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
model.to(DEVICE)


# --- Custom Dataset (Directly reads from folders) ---
class AudioDatasetFromFolders(Dataset):
    def __init__(self, audio_root, split_name, processor):
        """
        Args:
            audio_root (string): Base directory (e.g., 'data/raw_data').
            split_name (string): Name of the split folder (e.g., 'TRAIN', 'TEST', 'VALIDATION').
            processor (Wav2Vec2Processor): Wav2Vec2 processor for tokenization and feature extraction.
        """
        self.processor = processor
        self.data = [] # List to store (audio_path, label) tuples

        # Define expected subfolders for labels
        label_folders = {"SOBER": 0, "DRUNK": 1} # Mapping folder names to numerical labels

        split_path = os.path.join(audio_root, split_name)
        if not os.path.isdir(split_path):
            raise ValueError(f"Split folder '{split_path}' not found. Please check RAW_AUDIO_ROOT and split_name.")

        print(f"Scanning audio files in {split_path}...")
        initial_count = 0
        for folder_name, label_id in label_folders.items():
            label_path = os.path.join(split_path, folder_name)
            if not os.path.isdir(label_path):
                print(f"Warning: Label folder '{label_path}' not found. Skipping files from this label in {split_name}.")
                continue

            for filename in os.listdir(label_path):
                if filename.lower().endswith(".wav"):
                    full_audio_path = os.path.join(label_path, filename)
                    initial_count += 1
                    # Verify file existence before adding (optional, but good practice)
                    if os.path.exists(full_audio_path):
                        self.data.append((full_audio_path, label_id))
                    else:
                        print(f"Warning: Audio file not found: {full_audio_path}. Skipping.")
        
        if len(self.data) == 0:
            raise ValueError(f"No audio files found for '{split_name}' split in '{audio_root}'. Check folder structure and WAV files.")
        else:
            print(f"Found {len(self.data)} audio files for {split_name} split.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, label = self.data[idx]

        speech, sr = librosa.load(audio_path, sr=SAMPLING_RATE) # Ensure 16kHz
        return {"input_values": speech, "labels": label}

# --- Custom Collate Function for DataLoader ---
def collate_fn(batch):
    # 'processor' is now globally defined, so it's accessible here.
    input_features = [item["input_values"] for item in batch]
    labels = [item["labels"] for item in batch]

    batch = processor(input_features, sampling_rate=SAMPLING_RATE, padding=True, return_tensors="pt")
    batch["labels"] = torch.tensor(labels, dtype=torch.long)
    return batch

# --- Main Execution Block ---
# This entire block needs to be wrapped in if __name__ == '__main__': for multiprocessing on Windows.
if __name__ == '__main__':
    
    # --- Dataset and DataLoader Creation ---
    train_dataset = AudioDatasetFromFolders(RAW_AUDIO_ROOT, "TRAIN", processor)
    val_dataset = AudioDatasetFromFolders(RAW_AUDIO_ROOT, "VALIDATION", processor) 
    test_dataset = AudioDatasetFromFolders(RAW_AUDIO_ROOT, "TEST", processor)

    # Leveraging multiple CPU cores for faster data loading
    num_workers = os.cpu_count() // 2 
    # For Windows, num_workers > 0 requires the main execution block to be wrapped in if __name__ == '__main__':
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

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

            outputs = model(input_values, labels=labels)
            loss = outputs.loss

            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS 

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
                optimizer.step()
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
            model.save_pretrained("./best_wav2vec2_model")
            processor.save_pretrained("./best_wav2vec2_model")
            print(f"New best validation UAR: {best_val_uar:.4f}. Model saved!")

    # --- Testing Loop (after training) ---
    print("\n--- Running Test Set Evaluation ---")
    # Load the best model if it was saved (uncomment if you want to load from saved checkpoint)
    # model = Wav2Vec2ForSequenceClassification.from_pretrained("./best_wav2vec2_model", num_labels=NUM_LABELS)
    # processor = Wav2Vec2Processor.from_pretrained("./best_wav2vec2_model")
    # model.to(DEVICE) 

    model.eval()
    test_preds = []
    test_labels = []
    progress_bar_test = tqdm(test_dataloader, desc="Test Evaluation")

    for batch in progress_bar_test:
        input_values = batch["input_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        with torch.no_grad():
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
    print("You can find the best performing model (based on validation UAR) saved in './best_wav2vec2_model' directory.")
