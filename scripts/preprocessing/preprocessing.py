import librosa
import numpy as np
import soundfile as sf
import os
import matplotlib.pyplot as plt
import pandas as pd
import csv
from tqdm import tqdm


def split_audio(recording, sampling_rate, n_seconds, overlap):
    """
    Method to split a audio signal into pieces

    Input:
    recording   audio signal to split
    sampling_rate: The rate at which audio is sampled (samples per second)
    n_seconds: number of seconds in each split
    overlap: in seconds, overlap within splits
    
    Output: 
    List of split audio recordings
    """

    #If audio is in stereo, convert to mono
    if len(recording.shape) > 1:
        recording = librosa.to_mono(recording)

    if overlap >= n_seconds:
        raise Exception("Error: n_seconds <= overlap")

    def add_to_audio_list(y):
        if len(y) / sampling_rate < n_seconds:
            raise Exception(
                    f'Length of audio lesser than `split size in seconds` - {len(y) / sampling_rate} seconds, required {n_seconds} seconds')
        y = y[:required_length]
        audio_list.append(y)

    audio_list = []
    required_length = n_seconds * sampling_rate
    audio_in_seconds = len(recording) // sampling_rate

    # Check if the recording audio file is larger than the required number of seconds in a split
    if audio_in_seconds >= n_seconds:
        start = 0
        end = n_seconds
        left_out = None

        # Until highest multiple of n_seconds is reached, segment recording and store it in a list
        while end <= audio_in_seconds:
            index_at_start, index_at_end = start * sampling_rate, end * sampling_rate
            new_audio_sample = recording[index_at_start:index_at_end]
            add_to_audio_list(new_audio_sample)
            left_out = audio_in_seconds - end       #amount left unprocessed
            start = (start - overlap) + n_seconds   #updating the starting point
            end = (end - overlap) + n_seconds       #updating the ending point

        #For the remaining segment, the last n_seconds are added to the list
        if left_out > 0:
            new_audio_sample = recording[-n_seconds * sampling_rate:]
            add_to_audio_list(new_audio_sample)
    else:
        #If the recording is shorter, then the audio is repeated
        new_audio_sample = np.append(recording, recording)

        # If the recording is too short, it will be repeated multiple times
        while len(new_audio_sample) < (sampling_rate * n_seconds):
            new_audio_sample = np.hstack((new_audio_sample, new_audio_sample))
        add_to_audio_list(new_audio_sample)
    return audio_list

def mel_filters_with_spectrogram(audio,sampling_rate,filename):
    """
    Generates a spectrogram from the audio file


    Input:
    audio: audio recording
    sampling_rate
    filename: complete route where the spectrogram will be store
    """

    #Computing the mel spectrogram
    logmel = librosa.feature.melspectrogram(y=audio,n_mels=128,sr=sampling_rate)
    #Converting spectrogram into decibels scale
    logmel= librosa.power_to_db(logmel, ref=np.max)


    fig = plt.figure(figsize=(2.24, 2.24), dpi=100) # 2.24 inches * 100 dpi = 224 pixels
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    #Display spectrogram
    librosa.display.specshow(logmel, sr=sampling_rate, ax = ax, x_axis=None, y_axis=None, cmap='viridis')
    
    fig.savefig(filename, dpi=100, format = 'jpg')
    plt.close(fig)
    
def read_audio_basic_preprocess(filepath,file,label,base_path,image_path,sampling_rate,n_seconds,overlap):

    """
    Loads recording files, cut it into segments, resample them to 16kHz and generate spectrogram
    Input:
        file: name of recording file
        label
        base_path: folder with recordings
        sampling_rate: desired sampling rate of output
        n_seconds: length of each segment or split
        overlap: in seconds
    Output:
        out_data: list of spectrogram file names
        out_labels: list of labels
    """

    out_data, out_labels = [], []
    full_path=os.path.join(filepath, file)
    if os.path.exists(full_path):
        #Loading the recording file
        recording, sr = librosa.load(full_path,sr=None, mono=False)
        target_sr = sampling_rate

        #Splitting recording into segments of n_seconds length
        segments = split_audio(recording, sampling_rate=sr, n_seconds=n_seconds,
                           overlap=overlap)
        for i, segment in enumerate(segments):
            
            #Resampling to 16kHz
            segment_resampled = librosa.resample(segment, orig_sr = sr, target_sr = target_sr )

            #Generating name for spectrogram file
            out_filename = os.path.join(image_path, file)+"_"+str(i)+"_"+str(label)+'.jpg'
            #Generating the spectrogram of the segment audio
            mel_filters_with_spectrogram(segment_resampled, target_sr, out_filename)

            out_data.append(out_filename)
            out_labels.append(int(label))
            
    return out_data, out_labels

def preprocess_data(base_path,image_path,filepaths,files,labels,n_seconds,desired_sr,overlap):
    """
    Input:
    base_path: folder with audio files
    n_seconds: length of segments in seconds
    desired_sr: output sampling rate
    overlap: in seconds

    Output:
    out_data
    out_labels
    """

    out_data, out_labels = [],[]

    for filepath,file,label in tqdm(zip(filepaths,files,labels),total=len(labels)):
        out_preprocessing = read_audio_basic_preprocess(filepath,file,label,base_path,image_path,desired_sr,n_seconds,overlap)

        for i, out_label in enumerate(out_preprocessing[1]):
            out_data.append(out_preprocessing[0][i])
            out_labels.append(out_label)

    return out_data, out_labels

def start_preprocess(datalist_file,filename_to_save,shuffle=True):
    """
    Process a csv file with audio recording names and saves a csv file with spectrogram names

    Input:
    datalist_file: csv file with a list of filenames and drunk flag (A: drunk, N: not drunk)
    filename_to_save: name of the output csv file, this file has a list of the spectrogram file names and drunk flag 
    """
    #df[0]: file path
    #df[1]: files names
    #df[2]: labels (A = drunk, N = not drunk)
    df = pd.read_csv(datalist_file,header=None,delimiter=',')
    
    if shuffle:
        df=df.sample(frac=1)

    """
    Replacing Drunk Flag value:
        0 if N (not drunk)
        1 if A (drunk)
    """

    df[2]=df[2].apply(lambda x: 1 if x=='A' else 0)

    
    base_path=os.path.dirname(__file__)
    segment_seconds = 12
    desired_sr = 16000
    overlap = 0
    image_path = base_path

    #file= '0061006001_h_00.wav'
    #sf.write('0061006001_h_00_resampled.wav',y_resampled, 16000)
    out_data, out_labels = preprocess_data(base_path,image_path,df[0].values,df[1].values, df[2].values,segment_seconds,desired_sr,overlap)

    complete_output = np.concatenate((np.array([out_data]).T, np.array([out_labels]).T),axis=1)

    #Saving csv with list of spectrogram file routes and label
    columns=["spectrogram_fileP","label"]
    out_df = pd.DataFrame(complete_output,columns=columns) 
    out_filename = os.path.join(base_path, filename_to_save)
    out_df.to_csv(out_filename,index=False)



def classify_sessions(tbl_file):
    """
       Classifies sessions into drunk (BAK >=0.05%) or not drunk.
       Drunk flag:
        A: Drunk
        N: Not drunk

        Output structure:
       {SES####: 'A' or 'N'}
      
    """
    session_labels = {}
    with open(tbl_file, 'r', newline='') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            ses = row['SES']
            bak = float(row['BAK'])
            label = 'A' if bak >= 0.0005 else 'N'
            session_labels[f'ses{ses}'] = label
    return session_labels

def label_wav_files(root_folder, tbl_file, output_csv):
    """
    Generates a list of .wav files and assigns it a drunk flag (A drunk, N not drunk) base on BAK value per session

    Input:
    root_folder: path to rootfolder containing all of the sessions subfolders
    tbl_file: .tbl file with list of session numbers and bak levels per session


    Output:
    output_csv: .csv file (no header) with list of .wav files with drunk label 
    """

    #Going through all of the session subfolders and listing .wav files
    session_labels = classify_sessions(tbl_file)
    results = []
    for session_folder in os.listdir(root_folder):
        if session_folder in session_labels:
            label = session_labels[session_folder]
            folder_path = os.path.join(root_folder, session_folder)
            for file in os.listdir(folder_path):
                if file.endswith('.wav'):
                    results.append([folder_path,file, label])
    
    # Writing output list to CSV
    with open(output_csv, 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(results)


def process_tlb_subsetfile(tlb_path,output_csv):
    """
    Input:
    tlb_path: .tlb file with list of audio files and indicator of drunk(A)/sober(N)
    output_csv: path to store .csv file (no header) with list of .wav files with drunk label 
    """

    data=[]

    with open(tlb_path,'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                path, label = parts[0], parts[1] #Path, DrunkFlag

                #Extracting subfolder and filename
                path_parts = path.split('/')
                if len(path_parts) >= 2:
                    subfolder = path_parts[1].lower()
                    subfolder = os.path.join(os.path.dirname(__file__), subfolder)
                    filename = os.path.basename(path).lower()
                    data.append([subfolder,filename, label])

    # Writing output list to CSV
    with open(output_csv, 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(data)


#Generating csv with audio files list
#tbl_file = os.path.join(os.path.dirname(__file__), 'SESSEXT.TBL')
#root_folder = os.path.dirname(__file__)  
#output_csv = os.path.join(os.path.dirname(__file__), 'wav_labels.csv')

#label_wav_files(root_folder, tbl_file, output_csv)

#start_preprocess(output_csv,'spectrogram_list.csv',shuffle=True)

#Generating .csv file with list of training audio
tbl_file = os.path.join(os.path.dirname(__file__), 'TRAIN.TBL')
output_csv = os.path.join(os.path.dirname(__file__), 'wav_labelsTrain.csv')
process_tlb_subsetfile(tbl_file,output_csv)
start_preprocess(output_csv,'spectrogram_listTrain.csv',shuffle=True)

#Generating .csv file with list of test audio
tbl_file = os.path.join(os.path.dirname(__file__), 'TEST.TBL')
output_csv = os.path.join(os.path.dirname(__file__), 'wav_labelsTest.csv')
process_tlb_subsetfile(tbl_file,output_csv)
start_preprocess(output_csv,'spectrogram_listTest.csv',shuffle=True)


#Generating .csv file with list of validation audio
tbl_file = os.path.join(os.path.dirname(__file__), 'D1.TBL')
output_csv = os.path.join(os.path.dirname(__file__), 'wav_labelsVal.csv')
process_tlb_subsetfile(tbl_file,output_csv)
start_preprocess(output_csv,'spectrogram_listVal.csv',shuffle=True)