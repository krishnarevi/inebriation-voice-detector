import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
from datetime import datetime

# For Explainable AI
from captum.attr import IntegratedGradients
import IPython.display as ipd


# Suppress specific librosa warning about audioread backend
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
# Suppress the transformers warning about uninitialized weights in the classification head
warnings.filterwarnings(
    "ignore",
    message="Some weights of the model checkpoint at.*were not initialized.*",
    category=UserWarning,
    module="transformers"
)

# --- Configuration for XAI Script ---
TRAINED_MODEL_PATH = "/fast/krevi/v5_ext_split/run_20250622_183210/best_wav2vec2_model"
RAW_AUDIO_ROOT = "/fast/krevi/ALC_extended_split"
SAMPLING_RATE = 16000
NUM_LABELS = 2

# XAI explanation samples configuration
NUM_XAI_SAMPLES_PER_CLASS = 2 # Number of samples to explain for each class (e.g., 2 sober, 2 drunk)
# Total samples to explain will be NUM_XAI_SAMPLES_PER_CLASS * NUM_LABELS

OUTPUT_BASE_DIR = "xai_outputs"

# --- Global Initialization ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Custom Dataset (Re-using the definition without augmentation) ---
class AudioDatasetFromFolders(Dataset):
    def __init__(self, audio_root, split_name, processor):
        self.processor = processor
        self.data = []

        self.label_map = {"SOBER": 0, "DRUNK": 1} # SOBER is 0, DRUNK is 1

        split_path = os.path.join(audio_root, split_name)
        if not os.path.isdir(split_path):
            raise ValueError(f"Split folder '{split_path}' not found. Please check RAW_AUDIO_ROOT and split_name.")

        print(f"Scanning audio files in {split_path}...")
        for folder_name, label_id in self.label_map.items():
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, label = self.data[idx]
        speech, sr = librosa.load(audio_path, sr=SAMPLING_RATE)
        return {"input_values": speech, "labels": label, "audio_path": audio_path}

# --- Custom Collate Function for DataLoader ---
def collate_fn(batch):
    input_features = [item["input_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    audio_paths = [item["audio_path"] for item in batch]

    batch_processed = processor(input_features, sampling_rate=SAMPLING_RATE, padding=True, return_tensors="pt")
    batch_processed["labels"] = torch.tensor(labels, dtype=torch.long)
    batch_processed["audio_paths"] = audio_paths
    return batch_processed


if __name__ == '__main__':
    # --- Create Output Directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_BASE_DIR, f"xai_run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")

    # --- Load Model and Processor ---
    try:
        model = Wav2Vec2ForSequenceClassification.from_pretrained(TRAINED_MODEL_PATH, num_labels=NUM_LABELS)
        processor = Wav2Vec2Processor.from_pretrained(TRAINED_MODEL_PATH)
        model.to(DEVICE)
        model.eval() # Set model to evaluation mode
        print(f"Successfully loaded model from: {TRAINED_MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model from {TRAINED_MODEL_PATH}. Please ensure the path is correct and the model exists.")
        print(f"Error details: {e}")
        exit()

    # --- Prepare Test DataLoader ---
    test_dataset = AudioDatasetFromFolders(RAW_AUDIO_ROOT, "TEST", processor)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=os.cpu_count() // 2, pin_memory=True)
    print(f"Found {len(test_dataset)} samples in the test set.")

    # --- Run Inference and Collect Results ---
    print("\n--- Running Inference on Test Set to find best predictions ---")
    all_inference_results = [] # New list to store all inference details
    all_xai_candidate_results = [] # List for XAI candidates

    progress_bar_inference = tqdm(test_dataloader, desc="Inference on Test Set")
    for batch in progress_bar_inference:
        input_values = batch["input_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        audio_paths = batch["audio_paths"]

        with torch.no_grad(): # Use no_grad for the inference part
            outputs = model(input_values)
            
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)

        for i in range(len(labels)):
            true_label_id = labels[i].item()
            predicted_label_id = predictions[i].item()
            confidence_for_predicted_class = probabilities[i, predicted_label_id].item()

            # Prepare for full inference CSV
            all_inference_results.append({
                "audio_path": audio_paths[i],
                "ground_truth_id": true_label_id,
                "predicted_label_id": predicted_label_id,
                "confidence_predicted": confidence_for_predicted_class,
                "confidence_sober": probabilities[i, 0].item(), # Assuming 0 is SOBER
                "confidence_drunk": probabilities[i, 1].item()  # Assuming 1 is DRUNK
            })

            # Prepare for XAI selection
            if true_label_id == predicted_label_id: # Only consider correctly predicted samples for XAI
                all_xai_candidate_results.append({
                    "audio_path": audio_paths[i],
                    "true_label_id": true_label_id,
                    "predicted_label_id": predicted_label_id,
                    "confidence": confidence_for_predicted_class,
                    "input_values_tensor": input_values[i].cpu() # Store the CPU tensor
                })
    
    print("Inference complete.")

    # --- Save Full Inference Results to CSV ---
    full_inference_df = pd.DataFrame(all_inference_results)
    full_inference_csv_path = os.path.join(output_dir, "full_inference_results.csv")
    full_inference_df.to_csv(full_inference_csv_path, index=False)
    print(f"\nFull inference results for the test set saved to: {full_inference_csv_path}")


    # --- Identify Best Predicted Samples for XAI (Balanced by Class) ---
    sober_samples = [s for s in all_xai_candidate_results if s["true_label_id"] == 0]
    drunk_samples = [s for s in all_xai_candidate_results if s["true_label_id"] == 1]

    # Sort by confidence in descending order
    sober_samples.sort(key=lambda x: x["confidence"], reverse=True)
    drunk_samples.sort(key=lambda x: x["confidence"], reverse=True)

    # Select the top N for each class
    selected_xai_samples = []
    selected_xai_samples.extend(sober_samples[:NUM_XAI_SAMPLES_PER_CLASS])
    selected_xai_samples.extend(drunk_samples[:NUM_XAI_SAMPLES_PER_CLASS])

    if not selected_xai_samples:
        print("No correctly predicted samples found for XAI analysis based on the criteria. Cannot perform XAI.")
        exit()
    
    # Optional: Shuffle selected_xai_samples if you don't want them grouped by class in the output
    # import random
    # random.shuffle(selected_xai_samples)

    print(f"\n--- Analyzing {len(selected_xai_samples)} best predicted samples for XAI ({NUM_XAI_SAMPLES_PER_CLASS} per class) ---")

    # --- Explainable AI Section: Integrated Gradients ---

    # Define a predict function for Captum
    def predict_for_captum(input_tensor):
        return model(input_tensor).logits

    ig = IntegratedGradients(predict_for_captum)

    id_to_label = {v: k for k, v in test_dataset.label_map.items()}
    predictions_for_xai_csv = [] # Separate list for XAI specific samples

    for i, sample_info in enumerate(selected_xai_samples):
        audio_path = sample_info["audio_path"]
        true_label_id = sample_info["true_label_id"]
        predicted_label_id = sample_info["predicted_label_id"]
        confidence = sample_info["confidence"]
        
        true_label_name = id_to_label.get(true_label_id, f"Unknown({true_label_id})")
        predicted_label_name = id_to_label.get(predicted_label_id, f"Unknown({predicted_label_id})")

        print(f"\n--- Explaining Sample {i+1}/{len(selected_xai_samples)} ---")
        print(f"Audio Path: {audio_path}")
        print(f"True Label: {true_label_name} (ID: {true_label_id})")
        print(f"Predicted Label: {predicted_label_name} (ID: {predicted_label_id})")
        print(f"Prediction Confidence: {confidence:.4f}")

        predictions_for_xai_csv.append({
            "sample_id": i + 1,
            "audio_path": audio_path,
            "true_label": true_label_name,
            "predicted_label": predicted_label_name,
            "confidence": confidence
        })

        raw_speech, sr = librosa.load(audio_path, sr=SAMPLING_RATE)

        input_tensor_for_xai = sample_info["input_values_tensor"].unsqueeze(0).to(DEVICE)
        input_tensor_for_xai.requires_grad_(True) 

        baseline_tensor = torch.zeros_like(input_tensor_for_xai).to(DEVICE)

        # Compute attributions
        attributions_ig = ig.attribute(
            inputs=input_tensor_for_xai,
            baselines=baseline_tensor,
            target=predicted_label_id,
            return_convergence_delta=False,
        )

        attributions_np = attributions_ig.squeeze(0).cpu().detach().numpy() 

        processed_for_length = processor(raw_speech, sampling_rate=SAMPLING_RATE, return_tensors="pt").input_values
        actual_processed_length = processed_for_length.shape[1]

        fig, axs = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        
        time_original_audio = np.linspace(0, len(raw_speech) / SAMPLING_RATE, num=len(raw_speech))
        axs[0].plot(time_original_audio, raw_speech, color='blue', alpha=0.8)
        axs[0].set_title(f'Original Audio Waveform - True: {true_label_name}, Pred: {predicted_label_name} (Conf: {confidence:.2f})')
        axs[0].set_ylabel('Amplitude')
        
        times_processed_audio = np.linspace(0, actual_processed_length / SAMPLING_RATE, num=attributions_np.shape[0])
        axs[1].fill_between(times_processed_audio, 0, attributions_np, alpha=0.7, color='skyblue')
        axs[1].set_title('Integrated Gradients Attributions (Higher magnitude = more influence)')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Attribution')
        
        axs[0].set_xlim(0, actual_processed_length / SAMPLING_RATE)
        axs[1].set_xlim(0, actual_processed_length / SAMPLING_RATE)

        plt.tight_layout()
        
        audio_filename = os.path.basename(audio_path)
        plot_filename = f"sample_{i+1}_{predicted_label_name}_conf{confidence:.2f}_{os.path.splitext(audio_filename)[0]}.png"
        plt.savefig(os.path.join(output_dir, plot_filename), dpi=300)
        plt.close(fig)

        print("Playing audio (if in a compatible environment):")
        try:
            ipd.display(ipd.Audio(raw_speech, rate=SAMPLING_RATE))
        except Exception:
            print("Could not play audio (likely not in an interactive environment).")

        print("-" * 50)

    # Save details for the XAI-explained samples
    xai_samples_df = pd.DataFrame(predictions_for_xai_csv)
    xai_samples_csv_path = os.path.join(output_dir, "xai_explained_samples_details.csv")
    xai_samples_df.to_csv(xai_samples_csv_path, index=False)
    print(f"\nDetails of XAI-explained samples saved to: {xai_samples_csv_path}")

    print("\n--- Explainable AI Analysis and Output Saving Complete ---")