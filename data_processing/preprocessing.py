import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from tqdm import tqdm

# ✅ Tunable parameters (These serve as defaults for direct script execution)
SEGMENT_SECONDS_DEFAULT = 12
OVERLAP_SECONDS_DEFAULT = 0 
TARGET_SR_DEFAULT = 16000
N_MELS_DEFAULT = 224
IMG_SIZE_DEFAULT = (224, 224)
n_fft_DEFAULT = 1024
hop_length_DEFAULT = 256

# ✅ Input/output root folders
RAW_ROOT = r"D:\Uni\Lab\inebriation-voice-detector\data\raw_data"
PROCESSED_ROOT = r"D:\Uni\Lab\inebriation-voice-detector\data\processed\v1"

def split_audio(recording, sampling_rate, n_seconds, overlap):
    """
    Method to split an audio signal into pieces based on the user's specific requirements:
    1. Strictly non-overlapping n_seconds chunks.
    2. Any remaining partial segment (less than n_seconds) is included and padded with zeros to n_seconds.
    3. If the entire original audio is shorter than n_seconds, it's padded with zeros to exactly n_seconds.
    """
    if len(recording.shape) > 1:
        recording = librosa.to_mono(recording)
    
    # Overlap is effectively 0 for this specific splitting logic
    if overlap != 0:
        print(f"Warning: Overlap specified as {overlap}, but this splitting logic proceeds with strictly non-overlapping segments.")
    
    audio_list = []
    required_length_samples = int(n_seconds * sampling_rate)
    total_samples = len(recording)

    # --- Case 1: Audio is shorter than or exactly one segment length ---
    if total_samples <= required_length_samples:
        padding_needed = required_length_samples - total_samples
        padded_audio = np.pad(recording, (0, padding_needed), mode='constant')
        audio_list.append(padded_audio)
        return audio_list

    # --- Case 2: Audio is longer than one segment length ---
    start_sample = 0
    
    # Extract full, non-overlapping segments
    while (start_sample + required_length_samples) <= total_samples:
        segment = recording[start_sample : start_sample + required_length_samples]
        audio_list.append(segment)
        start_sample += required_length_samples # Move to the next non-overlapping position

    # Add the remaining partial segment, if any, and pad it to n_seconds
    if start_sample < total_samples:
        remaining_segment = recording[start_sample:]
        padding_needed = required_length_samples - len(remaining_segment)
        padded_remaining_segment = np.pad(remaining_segment, (0, padding_needed), mode='constant')
        audio_list.append(padded_remaining_segment)
        
    return audio_list

def mel_filters_with_spectrogram(audio, sampling_rate, output_path, n_mels, img_size, n_fft, hop_length):
    """
    Generates a log Mel spectrogram, applies a colormap, resizes, and saves it as a 3-channel image.
    and provides distinct RGB channels for models pre-trained on natural images.
    Parameters are now passed explicitly.
    """
    logmel = librosa.feature.melspectrogram(y=audio, sr=sampling_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    logmel_db = librosa.power_to_db(logmel, ref=np.max)

    # Normalize logmel_db to the range [0, 1] for colormap application
    norm_logmel_db = (logmel_db - logmel_db.min()) / (logmel_db.max() - logmel_db.min())
    cmap = cm.get_cmap('viridis')
    rgb_image = cmap(norm_logmel_db)[:, :, :3] # Take only the RGB channels, discard Alpha
    
    # Convert to 8-bit (0-255) pixel values
    img_array_255 = (rgb_image * 255).astype(np.uint8)

    # Convert numpy array to PIL Image
    img_pil = Image.fromarray(img_array_255)
    
    # Resize the image to the target size (e.g., 224x224) using bilinear interpolation
    img_pil = img_pil.resize(img_size, Image.BILINEAR) 
    
    # Save the 3-channel RGB image as JPEG
    img_pil.save(output_path, format='JPEG')

def process_audio_folder(split, segment_seconds, overlap_seconds, target_sr, n_mels, img_size, n_fft, hop_length):
    """
    Processes audio files in a given split folder, splits them into segments,
    generates spectrograms, and saves them. All parameters are adjustable.
    """
    input_base = os.path.join(RAW_ROOT, split)
    output_base = os.path.join(PROCESSED_ROOT, split) 
    labels = ['DRUNK', 'SOBER']

    total_spectrograms_in_split = 0

    for label in labels:
        input_folder = os.path.join(input_base, label)
        output_folder = os.path.join(output_base, label)
        os.makedirs(output_folder, exist_ok=True)

        folder_spectrogram_count = 0 

        for file in tqdm(os.listdir(input_folder), desc=f"{split}/{label}"):
            if not file.lower().endswith(".wav"):
                continue
            full_path = os.path.join(input_folder, file)
            try:
                y, sr = librosa.load(full_path, sr=None)
                # Pass segment_seconds and overlap_seconds to split_audio
                segments = split_audio(y, sr, segment_seconds, overlap_seconds) 
                for i, segment in enumerate(segments):
                    # Pass target_sr for resampling
                    resampled = librosa.resample(segment, orig_sr=sr, target_sr=target_sr)
                    filename = os.path.splitext(file)[0]
                    output_path = os.path.join(output_folder, f"{filename}_{i}.jpg")
                    # Pass all spectrogram generation parameters
                    mel_filters_with_spectrogram(resampled, target_sr, output_path, n_mels, img_size, n_fft, hop_length)
                    folder_spectrogram_count += 1
                    total_spectrograms_in_split += 1
            except Exception as e:
                print(f"Failed processing {file}: {e}")
        
        print(f"Finished processing {split}/{label}. Generated {folder_spectrogram_count} spectrograms.")
    
    return total_spectrograms_in_split

# === Master Preprocessing Pipeline ===
if __name__ == "__main__":
    # Define parameters for this specific run, using defaults or custom values
    current_segment_seconds = SEGMENT_SECONDS_DEFAULT
    current_overlap_seconds = OVERLAP_SECONDS_DEFAULT
    current_target_sr = TARGET_SR_DEFAULT
    current_n_mels = N_MELS_DEFAULT
    current_img_size = IMG_SIZE_DEFAULT
    current_n_fft = n_fft_DEFAULT
    current_hop_length = hop_length_DEFAULT

    grand_total_spectrograms = 0

    for split in ["TRAIN", "TEST", "VALIDATION"]:
        print(f"\n--- Processing {split} split ---")
        # Pass all parameters to process_audio_folder
        count_for_this_split = process_audio_folder(
            split, 
            current_segment_seconds, 
            current_overlap_seconds, 
            current_target_sr, 
            current_n_mels, 
            current_img_size, 
            current_n_fft, 
            current_hop_length
        )
        grand_total_spectrograms += count_for_this_split
        print(f"--- Total spectrograms for {split} split: {count_for_this_split} ---")

    print(f"\n=== Total spectrograms generated across all splits: {grand_total_spectrograms} ===")