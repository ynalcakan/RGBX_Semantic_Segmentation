import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Path to PST900 labels
label_dir = "datasets/PST900/Label_resized/"

# Create a colormap for better visualization
colors = [
    [0, 0, 0],        # background (black)
    [255, 0, 255],    # backpack (magenta)
    [0, 255, 0],      # fire extinguisher (green)
    [255, 0, 0],      # hand drill (red)
    [255, 255, 255],  # rescue randy (white)
]
colormap = np.array(colors, dtype=np.uint8)

# Get first few label files
label_files = sorted(os.listdir(label_dir))[:5]  # Let's look at first 5 images

plt.figure(figsize=(20, 8))
for idx, label_file in enumerate(label_files):
    # Read label image as grayscale
    label_path = os.path.join(label_dir, label_file)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    
    # Print unique values
    unique_values = np.unique(label)
    print(f"\nImage: {label_file}")
    print(f"Label values: {unique_values}")
    
    # Create colored visualization
    colored_label = colormap[label]
    
    # Display
    plt.subplot(2, 5, idx+1)
    plt.imshow(colored_label)
    plt.title(f'Label {idx+1}\nValues: {unique_values}')
    plt.axis('off')

plt.tight_layout()
plt.savefig('label_visualization.png')
plt.close()

print("\nVisualization saved as 'label_visualization.png'") 