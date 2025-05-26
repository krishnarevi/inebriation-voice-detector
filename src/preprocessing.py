
import pandas as pd
import librosa
import numpy as np
import soundfile as sf
import os
import matplotlib.pyplot as plt
import csv
import feature_extractor
import data_augmentation
import random 
from tqdm import tqdm
from pathlib import Path



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

    audio_list = []
    required_length = n_seconds * sampling_rate
    minimum_length = 0.25 * required_length

    def add_to_audio_list(y):
        #if len(y) / sampling_rate < n_seconds:
        #    raise Exception(
        #            f'Length of audio lesser than `split size in seconds` - {len(y) / sampling_rate} seconds, required {n_seconds} seconds')
        if len(y) < minimum_length: return
        if len(y) < required_length:
            # Pad with zeros at the end
            padding = required_length - len(y)
            y = np.pad(y, (0, padding), mode='constant')
        #else:
        #    y = y[:required_length]
        audio_list.append(y)

    audio_in_seconds = librosa.get_duration(y=recording, sr=sampling_rate)
    total_samples = len(recording)
    start_sample = 0
    step_size = int((n_seconds - overlap) * sampling_rate)

    while start_sample < total_samples:
        end_sample = start_sample + required_length
        segment = recording[start_sample:end_sample]
        add_to_audio_list(segment)
        start_sample += step_size

    #audio_in_seconds = len(recording) // sampling_rate
    # Check if the recording audio file is larger than the required number of seconds in a split
    #if audio_in_seconds >= n_seconds:
    #    start = 0
    #    end = n_seconds
    #    left_out = None

        # Until highest multiple of n_seconds is reached, segment recording and store it in a list
    #    while end <= audio_in_seconds:
    #        index_at_start, index_at_end = start * sampling_rate, end * sampling_rate
    #        new_audio_sample = recording[index_at_start:index_at_end]
    #        add_to_audio_list(new_audio_sample)
    #        left_out = audio_in_seconds - end       #amount left unprocessed
    #        start = (start - overlap) + n_seconds   #updating the starting point
    #        end = (end - overlap) + n_seconds       #updating the ending point

        #For the remaining segment, the last n_seconds are added to the list
    #    if left_out > 0:
    #        new_audio_sample = recording[-n_seconds * sampling_rate:]
    #        add_to_audio_list(new_audio_sample)
    #else:
        #If the recording is shorter, then the audio is repeated
    #    new_audio_sample = np.append(recording, recording)

        # If the recording is too short, it will be repeated multiple times
    #    while len(new_audio_sample) < (sampling_rate * n_seconds):
    #        new_audio_sample = np.hstack((new_audio_sample, new_audio_sample))
    #    add_to_audio_list(new_audio_sample)
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
    
def read_audio_basic_preprocess(filepath,file,label,image_path,sampling_rate,n_seconds,overlap,method=None):

    """
    Loads recording files, cut it into segments, resample them to 16kHz and generate spectrogram
    Input:
        file: name of recording file
        label
        sampling_rate: desired sampling rate of output
        n_seconds: length of each segment or split
        overlap: in seconds
    Output:
        out_data: list of spectrogram file names
        out_labels: list of labels
    """

    out_data, out_labels = [], []
    features_list = []
    method_flag = ''
    full_path=os.path.join(filepath, file)
    if os.path.exists(full_path):
        #Loading the recording file
        recording, sr = librosa.load(full_path,sr=None, mono=False)

        duration = librosa.get_duration(y=recording, sr=sr)
        rms = librosa.feature.rms(y=recording)[0]
        rms_mean = np.nanmean(rms)

        if method is not None:

            methods_list = ['shift','noise','pitch','masking']
            if duration <= 2.5:
                methods_list.remove('shift')
                methods_list.remove('masking')
            if rms_mean < 0.02 or rms_mean >0.08:
                methods_list.remove('noise')
            method = random.choice(methods_list)

        if method is not None:
            #Applying augmentation method
            if method == 'shift' and duration > 2.5:
                recording = data_augmentation.time_shift(recording, shift_max=0.2, sampling_rate=sr)
                method_flag = 'S'
            elif method == 'noise' and rms_mean >= 0.02 and rms_mean <=0.08:
                recording = data_augmentation.add_gaussian_noise(recording, noise_factor=0.005)
                method_flag = 'N'
            elif method == 'pitch':
                recording = data_augmentation.pitch_shift(recording, sampling_rate=sr)
                method_flag = 'P'

        target_sr = sampling_rate

        #Splitting recording into segments of n_seconds length
        segments = split_audio(recording, sampling_rate=sr, n_seconds=n_seconds,
                           overlap=overlap)
        for i, segment in enumerate(segments):
            
            if method == 'masking' and duration > 2.5:
                #Only apply time masking if the segment is longer than 2.5 seconds
                segment = data_augmentation.time_masking(segment)
                method_flag = 'M'

            #Resampling to 16kHz
            segment_resampled = librosa.resample(segment, orig_sr = sr, target_sr = target_sr )

            #Generating name for spectrogram file
            if label == 1:
                spectSave_folder = os.path.join(image_path,"DRUNK")
            else:
                spectSave_folder = os.path.join(image_path,"SOBER")
            #out_filename = os.path.join(image_path, file)+"_"+str(i)+"_"+str(label)+'.jpg'
            out_filename = os.path.join(spectSave_folder, file)+"_"+str(i)+method_flag+"_"+str(label)+'.jpg'
            #Extracting acoustic features from the segment
            features = feature_extractor.simple_acoustic_features(segment_resampled,target_sr,out_filename,label)
            features_list.append(features)

            #Generating the spectrogram of the segment audio
            mel_filters_with_spectrogram(segment_resampled, target_sr, out_filename) #UNDO

            out_data.append(out_filename)
            out_labels.append(int(label))
            
    return out_data, out_labels, features_list

def preprocess_data(image_path,filepaths,files,labels,mergeIDs,n_seconds,desired_sr,overlap,method=None):
    """
    Input:
    n_seconds: length of segments in seconds
    desired_sr: output sampling rate
    overlap: in seconds

    Output:
    out_data
    out_labels
    """

    out_data, out_labels, features_list = [],[], []

    for filepath,file,label,mergeID in tqdm(zip(filepaths,files,labels,mergeIDs),total=len(labels)):
        if method is not None:
            priority = priority_prob_map.get(mergeID, float('inf'))
            random_val = np.random.rand()
            if random_val > priority:
                continue
            
        out_preprocessing = read_audio_basic_preprocess(filepath,file,label,image_path,desired_sr,n_seconds,overlap,method)

        for i, out_label in enumerate(out_preprocessing[1]):
            out_data.append(out_preprocessing[0][i])
            out_labels.append(out_label)
            features_list.append(out_preprocessing[2][i])

    return out_data, out_labels, features_list

def start_preprocess(datalist_file,filename_to_save,filefeat_to_save,shuffle=True,method=None):
    """
    Process a csv file with audio recording names and saves a csv file with spectrogram names

    Input:
    datalist_file: csv file with a list of filenames and drunk flag (A: drunk, N: not drunk)
    filename_to_save: name of the output csv file, this file has a list of the spectrogram file names and drunk flag 
    filefeat_to_save: name of the output csv file, this file has a list of the acoustic features and drunk flag
    method: augmentation method to be used
    """
    #df[0]: file path
    #df[1]: files names
    #df[2]: labels (A = drunk, N = not drunk)
    df = pd.read_csv(datalist_file,header=None,delimiter=',',dtype={6: str})
    
    if shuffle:
        df=df.sample(frac=1)

    """
    Replacing Drunk Flag value:
        0 if N (not drunk)
        1 if A (drunk)
    """

    df[2]=df[2].apply(lambda x: 1 if x=='A' else 0)

    
    #base_path=os.path.dirname(__file__) #UNDO
    base_path = spect_folder
    segment_seconds = 12
    desired_sr = 16000
    overlap = 0
    image_path = spect_folder #UNDO base_path

    #file= '0061006001_h_00.wav'
    #sf.write('0061006001_h_00_resampled.wav',y_resampled, 16000)
    out_data, out_labels,features_list = preprocess_data(image_path,df[0].values,df[1].values, df[2].values,df[6].values,segment_seconds,desired_sr,overlap,method)

    complete_output = np.concatenate((np.array([out_data]).T, np.array([out_labels]).T),axis=1)

    #Saving csv with list of spectrogram file routes and label
    columns=["spectrogram_fileP","label"]
    out_df = pd.DataFrame(complete_output,columns=columns) 
    out_filename = os.path.join(base_path, filename_to_save)
    out_df.to_csv(out_filename,index=False)

    
    # Convert to DataFrame and save to CSV
    features_df = pd.DataFrame(features_list)

    # Move 'file_name' to the beginning
    cols = features_df.columns.tolist()
    cols = [cols[-2]] + cols[:-2] + [cols[-1]]  # file_name, rest..., label

    # Reorder DataFrame
    features_df = features_df[cols]
    out_filefeat = os.path.join(base_path, filefeat_to_save)
    features_df.to_csv(out_filefeat, index=False)


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

def bak_per_session(tbl_file):
    """
    Generates a dictionary with session numbers and BAK levels per session
    Input:
    tbl_file: .tbl file with list of session numbers and bak levels per session
    Output:
    data: dictionary with session numbers and BAK levels per session
    """
    sessions_data = {}
    with open(tbl_file, 'r', newline='') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            ses = 'ses'+row['SES']
            bak = float(row['BAK'])
            sessions_data[ses] = bak
    return sessions_data

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

    #Going through all of the session and getting its BAK values
    session_bac = bak_per_session(tbl_file)

    results = []
    for session_folder in os.listdir(root_folder):
        if session_folder in session_labels:
            label = session_labels[session_folder]
            folder_path = os.path.join(root_folder, session_folder)
            bac = session_bac[session_folder]
            for file in os.listdir(folder_path):
                if file.endswith('.wav'):
                    results.append([folder_path,file, label,bac])
    
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


def add_BACinfo(csv_file,bak_tbl_file,type):
    """
    Adds BAC information to the csv file with audio files list
    
    Input:
    csv_file: .csv file with list of audio files and indicator of drunk(A)/sober(N)

    Output:
    csv_file: .csv file with: list of audio files, indicator of drunk(A)/sober(N), BAC level and taskID
    """
    
    if type == 'WAV':
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file, header=None, delimiter=',')
        if df.columns.size != 3:
            return
        # Create a new DataFrame to store the results
        results = []
        bac_levels = bak_per_session(bak_tbl_file)
        # Iterate through each row in the original DataFrame
        for index, row in df.iterrows():
            file_path = row[0]
            file_name = row[1]
            label = row[2]

            # Extract the session and block number from the file path
            session_number = os.path.basename(file_path).split('_')[-1]
            block_number = session_number[3]  # Assuming the first character is the block number
            if block_number == '1' or block_number == '3':
                task_set = 'DRUNK'
            else:
                task_set = 'SOBER'

            #Extracting task ID
            task_id = file_name.split('_')[0][-3:]

            # Get the BAC level and task ID from the session number
            bac_level = bac_levels[session_number]

            # Append the results to the new DataFrame
            results.append([file_path, file_name, label, bac_level, task_set, task_id])

        # Convert the results to a DataFrame and save it to a new CSV file
        results_df = pd.DataFrame(results, columns=['file_path', 'file_name', 'label', 'BAC_level', 'task_set', 'task_id'])
        results_df.to_csv(csv_file, index=False, header=False)

    elif type == 'SPECT':  
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file, delimiter=',')
        if df.columns.size != 2:
            return
        
        # Create a new DataFrame to store the results
        results = []
        bac_levels = bak_per_session(bak_tbl_file)

        # Iterate through each row in the original DataFrame
        for index, row in df.iterrows():
            file_path = row[0]
            label = row[1]


            # Extract the session and block number from the file path
            file_name = os.path.basename(file_path).split('_')[0]
            session_number = file_name[-7:-3]
            task_id = file_name[-3:]
            block_number = session_number[0]
            session_number = 'ses' + session_number

            if block_number == '1' or block_number == '3':
                task_set = 'DRUNK'
            else:
                task_set = 'SOBER'

            # Get the BAC level and task ID from the session number
            bac_level = bac_levels[session_number]  

            # Append the results to the new DataFrame
            results.append([file_path, label, bac_level, task_set, task_id])  

        # Convert the results to a DataFrame and save it to a new CSV file
        results_df = pd.DataFrame(results, columns=['spectrogram_fileP', 'label', 'BAC_level', 'task_set', 'task_id'])
        results_df.to_csv(csv_file, index=False)


def add_MergeTaskID(csv_file,task_file):
    """
    Adds Merge task ID to the csv file with audio files list
    
    Input:
    csv_file: .csv file with list of audio files
    task_file: .xlsx file with task IDs

    Output:
    csv_file: .csv file with additional MergeTaskID
    """
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file, delimiter=',', dtype={'task_id': str})
    
    drunk_tasks = pd.read_excel(task_file, sheet_name='Drunk tasks',dtype={'Drunk task ID': str,'Sober task ID': str, 'Merge task ID': str})
    sober_tasks = pd.read_excel(task_file, sheet_name='Sober tasks',dtype={'Drunk task ID': str,'Sober task ID': str, 'Merge task ID': str})

    #Creating mapping dictionaries
    drunk_map = dict(zip(drunk_tasks['Drunk task ID'], drunk_tasks['Merge task ID']))
    sober_map = dict(zip(sober_tasks['Sober task ID'], sober_tasks['Merge task ID']))

    #Function to apply correct mapping acordinng to task set
    def map_merge_id(row):
        if row["task_set"] == 'DRUNK':
            return drunk_map.get(row["task_id"])
        elif row["task_set"] == 'SOBER':
            return sober_map.get(row["task_id"])
        else:
            return None
        
    df['merge_id'] = df.apply(map_merge_id, axis=1)
    df.to_csv(csv_file, index=False)

#Folder paths
audio_folder = Path(r"C:\Users\nagap\OneDrive\Documentos\Maestria\2025S\Phonetics TeamLab\ALC")
spect_folder = Path(r"C:\Users\nagap\OneDrive\Documentos\Maestria\2025S\Phonetics TeamLab\ALC\Spect")
list_folder = Path(r"C:\Users\nagap\OneDrive\Documentos\Maestria\2025S\Phonetics TeamLab\ALC")
task_folder = Path(r"C:\Users\nagap\OneDrive\Documentos\Maestria\2025S\Phonetics TeamLab\DrunkenLinguists\inebriation-voice-detector")
task_file = os.path.join(task_folder, 'TaskACL.xlsx')
bak_file = os.path.join(list_folder, 'SESSEXT.TBL')

#Generating csv with audio files list
#tbl_file = os.path.join(os.path.dirname(__file__), 'SESSEXT.TBL')
#root_folder = os.path.dirname(__file__)  
#output_csv = os.path.join(os.path.dirname(__file__), 'wav_labels.csv')

#label_wav_files(root_folder, tbl_file, output_csv)

#start_preprocess(output_csv,'spectrogram_list.csv',shuffle=True)

script_dir = Path(__file__).resolve().parent.parent
priority_folder = os.path.join(script_dir,'notebooks')
priority_file = 'priority_merge_ids.csv'
priority_fpath = os.path.join(priority_folder,priority_file)
priority_merge_ids = pd.read_csv(priority_fpath, dtype=str)['Merge ID'].str.strip()
max_rank = len(priority_merge_ids)
priority_prob_map = {
    mid: 1 - (rank / max_rank)  # higher rank = lower probability
    for rank, mid in enumerate(priority_merge_ids)
}

#Generating .csv file with list of training audio
#tbl_file = os.path.join(os.path.dirname(__file__), 'TRAIN.TBL')
output_csv = os.path.join(list_folder, 'wav_labelsTrainUpdated.csv')
#add_BACinfo(output_csv,bak_file,'WAV')
#process_tlb_subsetfile(tbl_file,output_csv)
#start_preprocess(output_csv,'spectrogram_listTrainA.csv','acoustic_featuresTrainA.csv',shuffle=True,method='Yes')
spect_csv = os.path.join(spect_folder, 'spectrogram_listTrainA.csv')
add_BACinfo(spect_csv,bak_file,'SPECT')
add_MergeTaskID(spect_csv,task_file)

#Generating .csv file with list of test audio
#tbl_file = os.path.join(os.path.dirname(__file__), 'TEST.TBL')
#output_csv = os.path.join(list_folder, 'wav_labelsTest.csv')
#add_BACinfo(output_csv,bak_file,'WAV')
#process_tlb_subsetfile(tbl_file,output_csv)
#start_preprocess(output_csv,'spectrogram_listTest.csv','acoustic_featuresTest.csv',shuffle=True,method='masking')
spect_csv = os.path.join(list_folder, 'spectrogram_listTest.csv')
#add_BACinfo(spect_csv,bak_file,'SPECT')
#add_MergeTaskID(spect_csv,task_file)

#Generating .csv file with list of validation audio
#tbl_file = os.path.join(os.path.dirname(__file__), 'D1.TBL')
#output_csv = os.path.join(list_folder, 'wav_labelsVal.csv')
#add_BACinfo(output_csv,bak_file,'WAV')
#process_tlb_subsetfile(tbl_file,output_csv)
#start_preprocess(output_csv,'spectrogram_listVal.csv','acoustic_featuresVal.csv',shuffle=True,method='masking')
spect_csv = os.path.join(list_folder, 'spectrogram_listVal.csv')
#add_BACinfo(spect_csv,bak_file,'SPECT')
#add_MergeTaskID(spect_csv,task_file)