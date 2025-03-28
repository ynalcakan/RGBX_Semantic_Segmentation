import os
import cv2
from tqdm import tqdm
import numpy as np

def resize_images(input_dir, output_dir, target_size=(640, 480), single_channel=False):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    
    print(f"Processing {len(image_files)} images from {input_dir}")
    
    # Process each image
    for img_file in tqdm(image_files):
        # Read image
        img_path = os.path.join(input_dir, img_file)
        if single_channel:
            # Read as grayscale for thermal and label images
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            # Read as RGB
            img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        # Resize image
        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Save resized image
        output_path = os.path.join(output_dir, img_file)
        cv2.imwrite(output_path, resized_img)

def main():
    # Base directory
    base_dir = "datasets/PST900"
    
    # RGB images (3 channels)
    rgb_input = os.path.join(base_dir, "RGB")
    rgb_output = os.path.join(base_dir, "RGB_resized")
    print("\nProcessing RGB images...")
    resize_images(rgb_input, rgb_output, single_channel=False)
    
    # Thermal images (1 channel)
    thermal_input = os.path.join(base_dir, "Thermal")
    thermal_output = os.path.join(base_dir, "Thermal_resized")
    print("\nProcessing Thermal images...")
    resize_images(thermal_input, thermal_output, single_channel=True)

    # Label images (1 channel)
    label_input = os.path.join(base_dir, "Label")
    label_output = os.path.join(base_dir, "Label_resized")
    print("\nProcessing Label images...")
    resize_images(label_input, label_output, single_channel=True)
    
    print("\nResizing complete!")

if __name__ == "__main__":
    main() 