import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

def detect_padding(img):
    """Detect if image has padding by checking for black regions around edges.
    Returns tuple of (has_padding, effective_size, padding_info)"""
    if len(img.shape) == 3:
        # For RGB images, convert to grayscale for padding detection
        check_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        check_img = img
    
    h, w = check_img.shape
    
    # Check top padding
    top = 0
    while top < h and np.all(check_img[top, :] == 0):
        top += 1
        
    # Check bottom padding
    bottom = h - 1
    while bottom >= 0 and np.all(check_img[bottom, :] == 0):
        bottom -= 1
        
    # Check left padding
    left = 0
    while left < w and np.all(check_img[:, left] == 0):
        left += 1
        
    # Check right padding
    right = w - 1
    while right >= 0 and np.all(check_img[:, right] == 0):
        right -= 1
    
    has_padding = (top > 0 or bottom < h-1 or left > 0 or right < w-1)
    effective_size = (bottom - top + 1, right - left + 1)
    padding_info = {
        'top': top,
        'bottom': h - bottom - 1,
        'left': left,
        'right': w - right - 1
    }
    
    return has_padding, effective_size, padding_info

def check_image_sizes(dataset_path):
    """Check for size mismatches and padding between RGB, thermal, and label images."""
    print(f"\nChecking image sizes and padding in {os.path.basename(dataset_path)}:")
    print("-" * 50)
    
    # Get paths for different modalities
    label_path = os.path.join(dataset_path, "Label")
    rgb_path = os.path.join(dataset_path, "RGB")
    thermal_path = os.path.join(dataset_path, "Thermal")
    
    # Get all files
    label_files = sorted(glob(os.path.join(label_path, "*.png")))
    
    if not label_files:
        print(f"No label files found in {label_path}")
        return
    
    mismatches = []
    sizes = {}
    padding_stats = {'rgb': [], 'thermal': [], 'label': []}
    
    # Process all files with progress bar
    for label_file in tqdm(label_files, desc="Checking image sizes"):
        base_name = os.path.basename(label_file)
        rgb_file = os.path.join(rgb_path, base_name)
        thermal_file = os.path.join(thermal_path, base_name)
        
        # Read images
        label_img = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
        rgb_img = cv2.imread(rgb_file)
        thermal_img = cv2.imread(thermal_file, cv2.IMREAD_GRAYSCALE)
        
        if label_img is None or rgb_img is None or thermal_img is None:
            print(f"Warning: Could not read one or more images for {base_name}")
            continue
        
        # Get sizes and check padding
        label_size = label_img.shape
        rgb_size = rgb_img.shape[:2]
        thermal_size = thermal_img.shape
        
        label_padding = detect_padding(label_img)
        rgb_padding = detect_padding(rgb_img)
        thermal_padding = detect_padding(thermal_img)
        
        # Store sizes and padding info
        sizes[base_name] = {
            'label': {'size': label_size, 'padding': label_padding},
            'rgb': {'size': rgb_size, 'padding': rgb_padding},
            'thermal': {'size': thermal_size, 'padding': thermal_padding}
        }
        
        # Track padding statistics
        if label_padding[0]: padding_stats['label'].append(base_name)
        if rgb_padding[0]: padding_stats['rgb'].append(base_name)
        if thermal_padding[0]: padding_stats['thermal'].append(base_name)
        
        # Check for mismatches in effective size
        if (label_padding[1] != rgb_padding[1] or 
            label_padding[1] != thermal_padding[1] or
            label_size != rgb_size or 
            label_size != thermal_size):
            mismatches.append({
                'file': base_name,
                'label': {'size': label_size, 'effective': label_padding[1], 'padding': label_padding[2]},
                'rgb': {'size': rgb_size, 'effective': rgb_padding[1], 'padding': rgb_padding[2]},
                'thermal': {'size': thermal_size, 'effective': thermal_padding[1], 'padding': thermal_padding[2]}
            })
    
    # Report findings
    print(f"\nTotal images checked: {len(label_files)}")
    print(f"Number of mismatches found: {len(mismatches)}")
    
    # Report padding statistics
    print("\nPadding Statistics:")
    for modality in ['label', 'rgb', 'thermal']:
        count = len(padding_stats[modality])
        if count > 0:
            print(f"\n{modality.upper()} images with padding: {count}/{len(label_files)} ({count/len(label_files)*100:.2f}%)")
    
    if mismatches:
        print("\nMismatched files:")
        for m in mismatches:
            print(f"\nFile: {m['file']}")
            for mod in ['label', 'rgb', 'thermal']:
                print(f"  {mod.upper()}:")
                print(f"    Full size: {m[mod]['size']}")
                print(f"    Effective size: {m[mod]['effective']}")
                print(f"    Padding: top={m[mod]['padding']['top']}, bottom={m[mod]['padding']['bottom']}, "
                      f"left={m[mod]['padding']['left']}, right={m[mod]['padding']['right']}")
    else:
        print("\nNo size or effective size mismatches found!")
    
    # Report unique sizes found
    unique_sizes = {}
    for file_data in sizes.values():
        for modality, data in file_data.items():
            if data['size'] not in unique_sizes:
                unique_sizes[data['size']] = {'count': 0, 'modalities': set()}
            unique_sizes[data['size']]['count'] += 1
            unique_sizes[data['size']]['modalities'].add(modality)
    
    print("\nUnique sizes found:")
    for size, data in unique_sizes.items():
        print(f"\nSize {size}:")
        print(f"  Count: {data['count']}")
        print(f"  Found in modalities: {', '.join(data['modalities'])}")

def analyze_dataset_labels(dataset_path):
    """Analyze labels from all images in a dataset."""
    print(f"\nAnalyzing {os.path.basename(dataset_path)} dataset:")
    print("-" * 50)
    
    # Get all label files
    label_files = sorted(glob(os.path.join(dataset_path, "*.png")))
    if not label_files:
        print(f"No label files found in {dataset_path}")
        return
    
    print(f"Total number of images: {len(label_files)}")
    
    all_unique_labels = set()
    label_stats = {}
    label_counts = {}  # To store total pixel counts per class
    
    # Process all files with progress bar
    for file_path in tqdm(label_files, desc="Processing images"):
        label = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            print(f"Warning: Could not read {file_path}")
            continue
            
        unique_labels = np.unique(label)
        all_unique_labels.update(unique_labels)
        
        # Count pixels per class
        total_pixels = label.size
        for lbl in unique_labels:
            count = np.sum(label == lbl)
            percentage = (count / total_pixels) * 100
            
            if lbl not in label_stats:
                label_stats[lbl] = []
                label_counts[lbl] = 0
            
            label_stats[lbl].append(percentage)
            label_counts[lbl] += count

    # Calculate and display statistics
    print("\nDataset Statistics:")
    print("-" * 50)
    print(f"All unique labels found: {sorted(list(all_unique_labels))}")
    print(f"Number of classes: {len(all_unique_labels)}")
    
    print("\nPer-class Statistics:")
    print("-" * 50)
    total_pixels_all = sum(label_counts.values())
    
    for label in sorted(label_stats.keys()):
        occurrences = len(label_stats[label])  # In how many images this label appears
        avg_percentage = np.mean(label_stats[label])
        std_percentage = np.std(label_stats[label])
        total_percentage = (label_counts[label] / total_pixels_all) * 100
        
        print(f"\nLabel {label}:")
        print(f"  - Appears in {occurrences}/{len(label_files)} images ({(occurrences/len(label_files))*100:.2f}%)")
        print(f"  - Average percentage per image: {avg_percentage:.2f}% (±{std_percentage:.2f}%)")
        print(f"  - Total percentage in dataset: {total_percentage:.2f}%")
        print(f"  - Total pixels: {label_counts[label]}")

def main():
    # Paths to both datasets
    mfnet_path = "datasets/MFNet"
    pst900_path = "datasets/PST900"
    
    # Check image sizes
    print("\n=== MFNet Dataset Size Check ===")
    check_image_sizes(mfnet_path)
    
    print("\n=== PST900 Dataset Size Check ===")
    check_image_sizes(pst900_path)
    
    # Analyze labels
    print("\n=== MFNet Dataset Label Analysis ===")
    analyze_dataset_labels(os.path.join(mfnet_path, "Label"))
    
    print("\n=== PST900 Dataset Label Analysis ===")
    analyze_dataset_labels(os.path.join(pst900_path, "Label_resized"))

if __name__ == "__main__":
    main() 