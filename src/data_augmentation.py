import numpy as np
import librosa
import random

def time_masking(audio, num_masks=1, mask_factor=0.1):
    """
    Apply time masking to an audio file.
    
    Parameters:
    - audio: The input audio signal.
    - sampling_rate: Sampling rate of the audio signal.
    - num_masks: Number of masks to apply.
    - mask_factor: Factor to determine the size of the mask.

    Outputs:
    - Masked audio signal.
    """
    masked_audio = audio.copy()
    total_samples = len(audio)
    mask_length = int(total_samples * mask_factor)
    if mask_length == 0:
        return masked_audio
    
    for _ in range(num_masks):
        start = np.random.randint(0, total_samples - mask_length)
        masked_audio[start:start + mask_length] = 0
    return masked_audio

def time_shift(audio, shift_max, sampling_rate):
    """
    Apply time shifting to the original audio signal.
    
    Parameters:
    - audio: The input audio signal.
    - shift_max: Maximum shift in seconds.
    - sampling_rate: Sampling rate of the audio signal.
    
    Returns:
    - Shifted audio signal.
    """
    shift = int(np.random.uniform(-shift_max, shift_max) * sampling_rate)
    if shift > 0:
        audio_shifted = np.pad(audio, (shift, 0), mode='constant')[:len(audio)]
    else:
        audio_shifted = np.pad(audio, (0, -shift), mode='constant')[-shift:]
    return audio_shifted

def add_gaussian_noise(audio, noise_factor=0.01):
    """
    Add Gaussian noise to the audio signal.
    
    Parameters:
    - audio: The input audio signal.
    - noise_factor: Factor to determine the amount of noise.
    
    Returns:
    - Audio signal with added Gaussian noise.
    """
    # Generate Gaussian noise
    noise = np.random.normal(0, noise_factor, audio.shape).astype(np.float32)
    
    #Add noise to the audio signal
    noisy_audio = audio +  noise_factor * noise

    noisy_audio = np.clip(noisy_audio, -1.0, 1.0)

    return noisy_audio

def pitch_shift(audio, sampling_rate):
    """
    Apply pitch shifting to the audio signal.
    
    Parameters:
    - audio: The input audio signal.
    - sampling_rate: Sampling rate of the audio signal.
    
    Returns:
    - Pitch-shifted audio signal.
    """

    n_steps = random.randint(-2, 2)
    return librosa.effects.pitch_shift(audio, sr=sampling_rate, n_steps=n_steps)