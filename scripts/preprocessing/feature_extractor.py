import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import librosa
import numpy as np
import parselmouth
import torchaudio
import torchaudio.transforms as T
import torch
import pandas as pd

def mfcc_features(audio, sampling_rate,normalize=False):
    """
    Extract MFCC features from an audio signal.
    
    Input:
    audio: Audio signal.
    sampling_rate: Sampling rate of the audio signal.
    normalize: Boolean flag to normalize the features.
    
    Output:
    np.ndarray: Extracted MFCC features.
    """
    mfccs = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=13)
    if normalize:
        mfccs_norm = np.mean(mfccs.T, axis=0) #average across time
        return mfccs_norm
    else:
        return mfccs
    
def extract_jitter_shimmer(audio, sampling_rate):
    """
    Extract jitter and shimmer features from an audio file.
    
    Input:
    file_path: Path to the audio file.
    """

    snd = parselmouth.Sound(audio,sampling_rate)
    point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)

    #Jitter
    local_jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    
    #Shimmer
    local_shimmer = parselmouth.praat.call([snd,point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3,1.6)

    return {
        "jitter": local_jitter,
        "shimmer": local_shimmer
    }

def extract_fbank_features(file_path, n_mels=80):
    waveform, sample_rate = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    

    mel_spec = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft = 400,
        win_length = 400,
        hop_length = 160,
        n_mels=n_mels,
        power = 2.0)(waveform)

    log_mel_spec = torch.log(mel_spec + 1e-6)  # Add small value to avoid log(0)
    return log_mel_spec.squeeze(0).T # Shape: (frames, n_mels)

def extract_features(audio, sampling_rate, normalize=True):
    """
    Basic acoustic features extractor.
    
    Input:
    file_path: Path to the audio file.
    
    Output:
    np.ndarray: Extracted MFCC features.
    """
        
    #Pitch (fundamental frequency) extraction
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sampling_rate)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0

    #Energy (RMS) "Loudness"
    energy = librosa.feature.rms(y=audio)[0]
    energy_mean = np.mean(energy)

    #Tempo (proxy for speaking rate)
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sampling_rate)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=13)
    if normalize:
        mfccs_norm = np.mean(mfccs.T, axis=0) #average across time
        

    #Spectral centroid (brightness)
    spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sampling_rate)
    spec_centroid_mean = np.mean(spec_centroid)

    return {
        "pitch_mean": pitch_mean,
        "energy_mean": energy_mean,
        "tempo": tempo,
        "spec_centroid_mean": spec_centroid_mean,
        **{f"mfcc_{i}": mfccs_norm[i] for i in range(13)}  
    }
    
    
def extract_acousticFeatures(listfile_csv, output_csv):
    """
    Extract acoustic features from a list of audio files and save to CSV.
    
    Input:
    listfile_csv: Path to the CSV file containing audio file paths and labels.
    """
    df = pd.read_csv(listfile_csv,header=None,delimiter=',')
    
    """
    Replacing Drunk Flag value:
        0 if N (not drunk)
        1 if A (drunk)
    """

    df[2]=df[2].apply(lambda x: 1 if x=='A' else 0)

    features_list = []
    
    for index, row in df.iterrows():
        file_path = row[0]
        file_name = row[1]
        full_file= os.path.join(file_path, file_name)
        label = row[2]
        
        
        # Load the audio file
        y, sr = librosa.load(full_file, sr=None)

        # Extract features
        features = extract_features(y, sr)
        jitter_shimmer = extract_jitter_shimmer(y, sr)
        #fbank_features = extract_fbank_features(full_file)

        if features is not None:
            features.update(jitter_shimmer)
            #features.update({"fbank_" + str(i): fbank_features[i] for i in range(fbank_features.shape[0])})
            features["label"] = label
            features_list.append(features)

    # Convert to DataFrame and save to CSV
    features_df = pd.DataFrame(features_list)
    features_df.to_csv(output_csv, index=False)    

def simple_acoustic_features(audio_file,sample_rate,file_name,label):
    
    features = []
    # Extract features
    features = extract_features(audio_file, sample_rate)
    jitter_shimmer = extract_jitter_shimmer(audio_file, sample_rate)

    if features is not None:
        features.update(jitter_shimmer)
        #features.update({"fbank_" + str(i): fbank_features[i] for i in range(fbank_features.shape[0])})
        features["file_name"] = file_name
        features["label"] = label
    return features  

listfile_csv = os.path.join(os.path.dirname(__file__), 'wav_labelsTrain2.csv')
output_csv = os.path.join(os.path.dirname(listfile_csv), 'acoustic_featuresTrain.csv')
extract_acousticFeatures(listfile_csv, output_csv)
