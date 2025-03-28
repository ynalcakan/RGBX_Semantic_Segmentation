import os
import cv2
import numpy as np

# Path to PST900 resized labels
label_dir = "datasets/PST900/Label_resized"

# Get all label files
label_files = sorted(os.listdir(label_dir))

# Track all unique values across all labels
all_unique_values = set()
value_counts = {}

print("Checking all label files...")
for label_file in label_files:
    label_path = os.path.join(label_dir, label_file)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    
    unique_values = np.unique(label)
    all_unique_values.update(unique_values)
    
    # Count occurrences of each value
    for val in unique_values:
        if val not in value_counts:
            value_counts[val] = 0
        value_counts[val] += 1
        
    # Print warning if unexpected values found
    unexpected = [v for v in unique_values if v > 4]
    if unexpected:
        print(f"File {label_file} has unexpected values: {unexpected}")

print("\nSummary:")
print("All unique values found across dataset:", sorted(list(all_unique_values)))
print("\nValue counts (number of images containing each value):")
for val in sorted(value_counts.keys()):
    print(f"Value {val}: found in {value_counts[val]} images") 