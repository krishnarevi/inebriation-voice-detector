import os
import shutil
import re
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Configuration ---
RAW_DATA_ROOT = r'D:\Uni\Lab\inebriation-voice-detector\data\raw_data'
ALC_AUDIO_DIR = os.path.join(RAW_DATA_ROOT, 'ALC')
SPLIT_TABLES_DIR = os.path.join(RAW_DATA_ROOT, 'split')
# New base directory for the new extended split, containing TRAIN, VALIDATION, TEST
OUTPUT_BASE_DIR = os.path.join(RAW_DATA_ROOT, 'ALC_split') 

# Ensure output directories exist
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE_DIR, 'TRAIN', 'SOBER'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE_DIR, 'TRAIN', 'DRUNK'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE_DIR, 'VALIDATION', 'SOBER'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE_DIR, 'VALIDATION', 'DRUNK'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE_DIR, 'TEST', 'SOBER'), exist_ok=True) # For test set sober
os.makedirs(os.path.join(OUTPUT_BASE_DIR, 'TEST', 'DRUNK'), exist_ok=True) # For test set drunk

# --- Helper Functions ---

def parse_sid_file(sid_file_path):
    """
    Parses SID.txt to get the mapping of ALC speaker ID (SCD) to their original SET.
    """
    sid_data = []
    try:
        with open(sid_file_path, 'r') as file:
            for line in file:
                # Skip comments or header lines if present (assuming they start with non-alphanumeric or #)
                if not line.strip() or line.strip().startswith('#') or line.strip().startswith('SET'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 2: # Ensure at least SET and SCD are present
                    s_set = parts[0]
                    scd = parts[1]
                    sid_data.append({'speaker_id': scd, 'original_set': s_set})
    except FileNotFoundError:
        print(f"Error: SID.txt not found at {sid_file_path}. This file is crucial for correct speaker mapping.")
        return pd.DataFrame()
    return pd.DataFrame(sid_data)

def parse_tbl_file(tbl_file_path):
    """
    Parses a .TBL file and returns a list of dictionaries with audio file info.
    Includes original file path, processed file name, and label.
    Uses 'A'/'N' (BAC threshold) for labeling as per user request.
    """
    data = []
    # Map for BAC-thresholded labels (column 2) - using this as per request
    bac_label_map = {'A': 'DRUNK', 'N': 'SOBER'} 

    try:
        with open(tbl_file_path, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')
                # Ensure there are at least 2 parts for original file path and challenge label
                if len(parts) >= 2: 
                    original_file_path = parts[0]
                    challenge_label = parts[1] # Column 2: 'A' or 'N' (BAC threshold)

                    # Remove BLOCK<number>/ prefix from the path, then normalize separators
                    # e.g., 'BLOCK10/SES1028/0271028001_h_00.WAV' becomes 'SES1028/0271028001_h_00.WAV'
                    file_name_clean = re.sub(r'^BLOCK\d+/', '', original_file_path, flags=re.IGNORECASE)
                    file_name_clean = os.path.normpath(file_name_clean) # Normalize slashes (e.g., / to \)
                    
                    # Determine the label to use: ONLY use 'A'/'N' based on column 2
                    label = bac_label_map.get(challenge_label) 
                    
                    if label: # Only process if we have a valid label ('A' or 'N')
                        # Extract speaker ID (SCD) from the numerical part of the filename.
                        # Assuming format like 'SSSUUUUUUU_X_DD.WAV' where SSS is 3-digit speaker ID
                        file_basename = os.path.basename(file_name_clean)
                        speaker_id_match = re.search(r'^(\d{3})\d+_[a-z]_(\d+)\.wav$', file_basename, re.IGNORECASE)
                        
                        speaker_id = speaker_id_match.group(1) if speaker_id_match else 'UNKNOWN' # Extract first 3 digits as speaker ID

                        data.append({
                            'original_tbl_path': original_file_path,
                            'file_name': file_name_clean, # This is now the path relative to ALC_AUDIO_DIR
                            'label': label,
                            'speaker_id': speaker_id,
                            'source_tbl': os.path.basename(tbl_file_path).replace('.TBL', '')
                        })
    except FileNotFoundError:
        print(f"Warning: TBL file not found at {tbl_file_path}. Skipping.")
    return data

def process_and_split_data():
    """
    Processes all .TBL files, combines data from TRAIN.TBL and D1.TBL for an extended train/val pool,
    performs an 80:20 stratified split on speakers in this pool, and prepares the TEST set from TEST.TBL.
    Ensures speaker disjunction based on SID.txt. Creates new folder structure and CSV.
    """
    print("--- Starting Data Preparation ---")

    # 0. Parse SID.txt to get the master speaker set mapping
    sid_df = parse_sid_file(os.path.join(SPLIT_TABLES_DIR, 'SID.txt'))
    if sid_df.empty:
        print("SID.txt could not be parsed or is empty. Cannot proceed with speaker-disjoint splitting.")
        return # Exit if SID.txt is crucial and not available

    # 1. Parse all TBL files
    train_data_raw = parse_tbl_file(os.path.join(SPLIT_TABLES_DIR, 'TRAIN.TBL'))
    d1_data_raw = parse_tbl_file(os.path.join(SPLIT_TABLES_DIR, 'D1.TBL'))
    test_data_raw = parse_tbl_file(os.path.join(SPLIT_TABLES_DIR, 'TEST.TBL'))
    # D2.TBL contains additional sober recordings for speaker-dependent tests,
    # which are ignored for this speaker-independent split.

    print(f"Parsed TRAIN.TBL: {len(train_data_raw)} entries")
    print(f"Parsed D1.TBL: {len(d1_data_raw)} entries")
    print(f"Parsed TEST.TBL: {len(test_data_raw)} entries")

    # Convert to DataFrames
    train_df_raw = pd.DataFrame(train_data_raw)
    d1_df_raw = pd.DataFrame(d1_data_raw)
    test_df_raw = pd.DataFrame(test_data_raw)

    # Merge with SID.txt to get the original_set for each speaker
    # Ensure speaker_id column is consistent type before merging
    sid_df['speaker_id'] = sid_df['speaker_id'].astype(str)
    
    if not train_df_raw.empty:
        train_df_raw = pd.merge(train_df_raw, sid_df, on='speaker_id', how='left')
    if not d1_df_raw.empty:
        d1_df_raw = pd.merge(d1_df_raw, sid_df, on='speaker_id', how='left')
    if not test_df_raw.empty:
        test_df_raw = pd.merge(test_df_raw, sid_df, on='speaker_id', how='left')
    
    # Check for unmapped speakers and warn
    unmapped_train = train_df_raw[train_df_raw['original_set'].isnull()]['speaker_id'].unique()
    if len(unmapped_train) > 0:
        print(f"WARNING: Speakers in TRAIN.TBL not found in SID.txt: {unmapped_train}")
    unmapped_d1 = d1_df_raw[d1_df_raw['original_set'].isnull()]['speaker_id'].unique()
    if len(unmapped_d1) > 0:
        print(f"WARNING: Speakers in D1.TBL not found in SID.txt: {unmapped_d1}")
    unmapped_test = test_df_raw[test_df_raw['original_set'].isnull()]['speaker_id'].unique()
    if len(unmapped_test) > 0:
        print(f"WARNING: Speakers in TEST.TBL not found in SID.txt: {unmapped_test}")
    
    # Remove any rows where original_set could not be determined (speaker ID not in SID.txt)
    train_df_raw.dropna(subset=['original_set'], inplace=True)
    d1_df_raw.dropna(subset=['original_set'], inplace=True)
    test_df_raw.dropna(subset=['original_set'], inplace=True)


    # 2. Define the fixed TEST set based on speakers from original D2_TEST set
    test_set_df = test_df_raw[test_df_raw['original_set'] == 'D2_TEST'].copy()
    test_set_df['Split'] = 'TEST'
    print(f"Test set prepared from D2_TEST speakers: {len(test_set_df)} samples.")
    
    # Get speakers from the D2_TEST set to ensure they are excluded from the train/val pool
    test_set_speakers = test_set_df['speaker_id'].unique().tolist()


    # 3. Combine TRAIN and D1 data for the new extended training/validation pool
    # Filter out any speakers that might accidentally appear in the test set (though should be disjunctive)
    extended_train_val_pool_df = pd.concat([train_df_raw, d1_df_raw], ignore_index=True)
    extended_train_val_pool_df = extended_train_val_pool_df[
        ~extended_train_val_pool_df['speaker_id'].isin(test_set_speakers)
    ].copy() # Ensure no test speakers are in the pool
    
    print(f"Combined TRAIN + D1 for extended pool (excluding test speakers): {len(extended_train_val_pool_df)} samples.")

    # 4. Perform 80:20 stratified split on the extended pool (speaker level)
    
    # Determine the stratification key for each speaker based on their label distribution
    # This accounts for speakers having both SOBER and DRUNK recordings
    speaker_label_distribution_pool = extended_train_val_pool_df.groupby('speaker_id')['label'].value_counts(normalize=True).unstack(fill_value=0)
    speaker_label_distribution_pool['stratify_key'] = speaker_label_distribution_pool.apply(
        lambda row: 'SOBER_DRUNK' if row.get('SOBER', 0) > 0 and row.get('DRUNK', 0) > 0 else ('SOBER' if row.get('SOBER', 0) > 0 else 'DRUNK'), axis=1
    )
    
    all_pool_speakers = extended_train_val_pool_df['speaker_id'].unique().tolist()
    
    speakers_for_splitting = []
    stratify_keys_for_splitting = []
    speakers_assigned_to_train_directly = [] # Speakers from strata with only one member

    # Iterate through all unique speakers in the combined pool
    for speaker_id in all_pool_speakers:
        key = speaker_label_distribution_pool.loc[speaker_id]['stratify_key']
        # Find all speakers that belong to this *same* stratification key in the current pool
        speakers_in_this_key_stratum = speaker_label_distribution_pool[
            speaker_label_distribution_pool['stratify_key'] == key
        ].index.tolist()
        
        # Check if this speaker_id is the ONLY one in its stratification key within the pool
        if len(speakers_in_this_key_stratum) == 1:
            # Assign this unique speaker to the training set directly
            if speaker_id not in speakers_assigned_to_train_directly: # Avoid duplicates if iterating over keys
                speakers_assigned_to_train_directly.append(speaker_id)
                print(f"  Assigning speaker '{speaker_id}' to TRAIN directly (single speaker in '{key}' stratum in pool).")
        else:
            # Add to the pool for stratified splitting
            speakers_for_splitting.append(speaker_id)
            stratify_keys_for_splitting.append(key)

    # Perform stratified split only on the speakers that can be split
    train_speakers_split = []
    val_speakers_split = []
    if speakers_for_splitting: # Only attempt split if there are speakers in the pool
        train_speakers_split, val_speakers_split = train_test_split(
            speakers_for_splitting,
            test_size=0.20, # 20% of speakers from this pool for new validation set
            stratify=stratify_keys_for_splitting, # Stratify by speaker's label profile
            random_state=42 # for reproducibility
        )
    
    # Combine directly assigned speakers with those from the stratified split for the final TRAIN
    train_speakers = speakers_assigned_to_train_directly + train_speakers_split
    # VALIDATION will only contain speakers from the stratified split
    val_speakers = val_speakers_split 

    # Filter original combined DataFrame to create the new train and val sets
    train_df = extended_train_val_pool_df[extended_train_val_pool_df['speaker_id'].isin(train_speakers)].copy()
    validation_df = extended_train_val_pool_df[extended_train_val_pool_df['speaker_id'].isin(val_speakers)].copy()

    train_df['Split'] = 'TRAIN'
    validation_df['Split'] = 'VALIDATION'

    print(f"New TRAIN set (from {len(train_speakers)} speakers): {len(train_df)} samples.")
    print(f"New VALIDATION set (from {len(val_speakers)} speakers): {len(validation_df)} samples.")
    
    # --- Verification of Speaker Disjunction (Crucial for Speaker-Independent Challenge) ---
    train_val_overlap = set(train_df['speaker_id']).intersection(set(validation_df['speaker_id']))
    if train_val_overlap:
        print(f"WARNING: Speaker overlap between TRAIN and VALIDATION: {train_val_overlap}")
    
    # Verify no speaker overlap with the fixed TEST set
    train_test_overlap = set(train_df['speaker_id']).intersection(set(test_set_df['speaker_id']))
    val_test_overlap = set(validation_df['speaker_id']).intersection(set(test_set_df['speaker_id']))
    if train_test_overlap or val_test_overlap:
        print(f"CRITICAL WARNING: Speaker overlap detected: Train-Test {train_test_overlap}, Val-Test {val_test_overlap}")
        print("This indicates a violation of speaker independence if original sets (TRAIN, D1, D2_TEST) are truly disjunctive by speakers.")
        print("Please review SID.txt parsing and original_set assignment.")


    # 5. Combine all dataframes for the final CSV
    final_df = pd.concat([train_df, validation_df, test_set_df], ignore_index=True)
    
    # Select desired columns for the CSV
    final_df = final_df[['file_name', 'Split', 'label', 'speaker_id']]
    final_df.rename(columns={'label': 'Label', 'file_name': 'FileName'}, inplace=True)

    # 6. Copy files to new directory structure
    # Use the combined DataFrame to iterate and copy
    files_to_copy_df = pd.concat([train_df, validation_df, test_set_df], ignore_index=True)
    
    print("\n--- Copying files to new split directories ---")
    for index, row in files_to_copy_df.iterrows():
        # 'file_name' contains the path relative to ALC_AUDIO_DIR (e.g., 'SES1028/file.wav')
        src_path = os.path.join(ALC_AUDIO_DIR, row['file_name']) 
        
        # Determine destination directory based on the 'Split' and 'label'
        dst_dir = os.path.join(OUTPUT_BASE_DIR, row['Split'], row['label']) 
        
        # Make sure the target label directory exists
        os.makedirs(dst_dir, exist_ok=True)
        
        # Use the base file name for the destination path (e.g., 'file.wav'), 
        # as the nested structure (SES1028/) is flattened into the class subfolder
        dst_path = os.path.join(dst_dir, os.path.basename(row['file_name']))

        if os.path.exists(src_path):
            try:
                shutil.copy2(src_path, dst_path)
                # print(f"Copied: {os.path.basename(src_path)} to {row['Split']}/{row['label']}")
            except Exception as e:
                print(f"Error copying {src_path} to {dst_path}: {e}")
        else:
            print(f"Missing source file: {src_path} (Skipping copy for this entry)")
    
    # 7. Save the CSV
    csv_output_path = os.path.join(OUTPUT_BASE_DIR, 'dataset_split.csv')
    final_df.to_csv(csv_output_path, index=False)
    print(f"\nCSV file generated at: {csv_output_path}")
    print("Data preparation complete.")

if __name__ == '__main__':
    process_and_split_data()
