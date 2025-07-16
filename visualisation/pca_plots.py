import os
import torch
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl # Import mpl for rcParams
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
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

# --- YOUR SCIENTIFIC PLOT STYLE CONFIGURATION ---
sns.set_theme(style="whitegrid") # Using whitegrid as per your bar plot, but will disable grid specifically for PCA plots below
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 22,
    "axes.titlesize": 26,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.dpi": 300,
    "axes.linewidth": 2,
    "lines.linewidth": 3,
    "lines.markersize": 10, # This might apply to scatter points as well, but 's' in scatterplot overrides it
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Define your dusty pastel colors if you want to use them for the PCA points' hue as well
# dusty_purple = '#B19CD9'
# dusty_blue = '#7DA7D9'

# --- Configuration (MUST MATCH your training config) ---
RAW_AUDIO_ROOT = r"D:\Uni\Lab\inebriation-voice-detector\data\raw_data\ALC_extended_split"
# UPDATE THIS PATH to your actual saved model directory
# Example: "/fast/krevi/v5_ext_split/run_20240101_123456/best_wav2vec2_model"
PATH_TO_SAVED_MODEL = r"D:\Uni\Lab\model\v5_ext_split\v5_ext_split\run_20250622_183210\best_wav2vec2_model" # <--- IMPORTANT: SET YOUR ACTUAL PATH HERE

MODEL_NAME_PRETRAINED = "jonatasgrosman/wav2vec2-large-xlsr-53-german" # Base pre-trained model for "before" plot
NUM_LABELS = 2 # Sober (0), Drunk (1)
SAMPLING_RATE = 16000 # Wav2Vec2 models are typically trained on 16kHz audio
BATCH_SIZE_PCA = 8 # Can be different from training batch size, adjust for memory

# --- Global Initialization ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Initialize processor for both models (should be consistent)
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME_PRETRAINED)

# --- Custom Dataset (Re-use from your code) ---
class AudioDatasetFromFolders(Dataset):
    def __init__(self, audio_root, split_name, processor, sampling_rate):
        self.processor = processor
        self.sampling_rate = sampling_rate
        self.data = []

        label_folders = {"SOBER": 0, "DRUNK": 1} # Ensure these match your actual labels

        split_path = os.path.join(audio_root, split_name)
        if not os.path.isdir(split_path):
            raise ValueError(f"Split folder '{split_path}' not found. Please check RAW_AUDIO_ROOT and split_name.")

        print(f"Scanning audio files in {split_path} for PCA...")
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, label = self.data[idx]
        speech, sr = librosa.load(audio_path, sr=self.sampling_rate) # Use self.sampling_rate
        return {"input_values": speech, "labels": label}

# --- Custom Collate Function (Re-use from your code) ---
def collate_fn(batch):
    input_features = [item["input_values"] for item in batch]
    labels = [item["labels"] for item in batch]

    batch = processor(input_features, sampling_rate=SAMPLING_RATE, padding=True, return_tensors="pt")
    batch["labels"] = torch.tensor(labels, dtype=torch.long)
    return batch

# --- Feature Extraction Function (Re-use) ---
def extract_hidden_states(model, dataloader, device):
    model.eval() # Set model to evaluation mode
    all_hidden_states = []
    all_labels = []

    print(f"Extracting hidden states from {len(dataloader.dataset)} samples...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            # Get the output from the base Wav2Vec2 model, not the classification head
            # output_hidden_states=True ensures we get all hidden states, but we only need last_hidden_state
            outputs = model.wav2vec2(input_values, output_hidden_states=True)
            last_hidden_states = outputs.last_hidden_state # (batch_size, sequence_length, hidden_size)

            # Average pooling across the sequence dimension for a single vector per sample
            pooled_hidden_states = last_hidden_states.mean(dim=1) # (batch_size, hidden_size)

            all_hidden_states.append(pooled_hidden_states.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.vstack(all_hidden_states), np.concatenate(all_labels)

# --- PCA Plotting Function (MODIFIED for scientific style) ---
def plot_pca(features, labels, title, filename, save_dir="."):
    """
    Performs PCA and plots the first two components with scientific styling.
    :param features: NumPy array of extracted features.
    :param labels: NumPy array of corresponding labels.
    :param title: Title for the plot.
    :param filename: Name of the file to save the plot (e.g., 'pca_plot.png').
    :param save_dir: Directory to save the plot. Defaults to current directory.
    """
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(features)

    df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
    df['Label'] = labels # Ensure your labels are 0 and 1 or adjust accordingly

    plt.figure(figsize=(10, 8)) # Adjusted for a scientific figure size
    sns.scatterplot(
        x='Principal Component 1',
        y='Principal Component 2',
        hue='Label',
        data=df,
        palette='viridis', # 'viridis' for distinct colors. Use `palette=[dusty_purple, dusty_blue]` if you want that specific pastel range
        s=40, # Increased marker size for better visibility in scientific plots
        alpha=0.7 # Add transparency if many points overlap
    )
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Label', frameon=False) # Remove legend frame as per your example
    plt.grid(False) # Explicitly remove grid lines
    plt.tight_layout()

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path_png = os.path.join(save_dir, filename)
    save_path_pdf = os.path.join(save_dir, filename.replace('.png', '.pdf')) # Save as PDF too

    plt.savefig(save_path_png)
    plt.savefig(save_path_pdf)
    print(f"PCA plot saved to: {os.path.abspath(save_path_png)} and {os.path.abspath(save_path_pdf)}")
    plt.close() # Close the plot to free memory


# --- Main Execution for Plotting ---
if __name__ == '__main__':
    # Define a directory to save the plots
    # This creates a 'pca_visuals' subfolder relative to where your saved model is.
    PLOTS_OUTPUT_DIR = os.path.join(os.path.dirname(PATH_TO_SAVED_MODEL), "pca_visuals_scientific")
    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
    print(f"PCA plots will be saved in: {PLOTS_OUTPUT_DIR}")

    # --- 1. Load the dataset for feature extraction ---
    test_dataset = AudioDatasetFromFolders(RAW_AUDIO_ROOT, "TEST", processor, SAMPLING_RATE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE_PCA, shuffle=False, collate_fn=collate_fn, num_workers=os.cpu_count() // 2, pin_memory=True)
    print(f"Using {len(test_dataset)} samples from the TEST set for PCA visualization.")

    # --- 2. Load the PRE-TRAINED model (for "before" plot) ---
    print("\nLoading pre-trained model for 'before fine-tuning' features...")
    model_pretrained = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME_PRETRAINED, num_labels=NUM_LABELS)
    model_pretrained.to(DEVICE)

    # --- 3. Extract features and plot PCA for "BEFORE" fine-tuning ---
    pre_finetune_features, pre_finetune_labels = extract_hidden_states(model_pretrained, test_dataloader, DEVICE)
    plot_pca(pre_finetune_features, pre_finetune_labels,
             '(a) PCA of Pre-trained Wav2Vec2.0 Features', 'pca_pretrained_wav2vec.png', PLOTS_OUTPUT_DIR)
    del model_pretrained # Free up memory
    torch.cuda.empty_cache() # Clear CUDA cache

    # --- 4. Load the FINE-TUNED model (for "after" plot) ---
    print(f"\nLoading fine-tuned model from '{PATH_TO_SAVED_MODEL}' for 'after fine-tuning' features...")
    try:
        model_finetuned = Wav2Vec2ForSequenceClassification.from_pretrained(PATH_TO_SAVED_MODEL, num_labels=NUM_LABELS)
    except Exception as e:
        print(f"ERROR: Could not load fine-tuned model from '{PATH_TO_SAVED_MODEL}'.")
        print(f"Please check the path and ensure the model was saved correctly. Error: {e}")
        exit() # Exit if the fine-tuned model cannot be loaded

    model_finetuned.to(DEVICE)

    # --- 5. Extract features and plot PCA for "AFTER" fine-tuning ---
    post_finetune_features, post_finetune_labels = extract_hidden_states(model_finetuned, test_dataloader, DEVICE)
    plot_pca(post_finetune_features, post_finetune_labels,
             '(b) PCA of Fine-tuned Wav2Vec2.0 Features', 'pca_fine_tuned_wav2vec.png', PLOTS_OUTPUT_DIR)
    del model_finetuned # Free up memory
    torch.cuda.empty_cache() # Clear CUDA cache

    print("\n--- PCA Visualization Complete ---")
    print(f"Check the '{PLOTS_OUTPUT_DIR}' directory for your plots.")