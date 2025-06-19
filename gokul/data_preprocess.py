import re

# Define the path to your .TBL file
tbl_file_path = r'D:\Uni\Lab\inebriation-voice-detector\data\raw_data\split\TRAIN.TBL'

# List to hold the processed results
data = []

# Read and process the file
with open(tbl_file_path, 'r') as file:
    for line in file:
        # Strip whitespace and split by tab
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            file_path = parts[0]
            label_code = parts[1]

            # Remove "BLOCK<number>/" using regex
            file_path = re.sub(r'^BLOCK\d+/', '', file_path, flags=re.IGNORECASE)

            # Map the label
            if label_code == 'A':
                label = 'drunk'
            elif label_code == 'N':
                label = 'sober'
            else:
                continue  # Skip unknown labels

            data.append((file_path, label))

# Output result
for file_path, label in data:
    print(f"{file_path}\t{label}")

