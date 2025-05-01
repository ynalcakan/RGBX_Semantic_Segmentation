import numpy as np
import cv2
import random


def temperature_scaling(img, scale_range=(0.8, 1.2)):
    """
    Scale the thermal image intensity to simulate different temperature conditions.
    
    Args:
        img: Input thermal image (numpy array)
        scale_range: Range of scaling factors (tuple)
        
    Returns:
        Augmented thermal image
    """
    # Preserve original dtype
    original_dtype = img.dtype
    
    # Convert to float32 for processing
    if not np.issubdtype(img.dtype, np.floating):
        img_float = img.astype(np.float32)
    else:
        img_float = img.copy()
    
    scale_factor = random.uniform(scale_range[0], scale_range[1])
    scaled_img = img_float * scale_factor
    
    # Clip values if needed
    if np.issubdtype(original_dtype, np.integer):
        scaled_img = np.clip(scaled_img, 0, 255)
    
    # Convert back to original dtype
    if scaled_img.dtype != original_dtype:
        scaled_img = scaled_img.astype(original_dtype)
    
    return scaled_img

def add_thermal_noise(img, noise_type='gaussian', severity=(10, 25)):
    """
    Add thermal sensor noise to the image.
    
    Args:
        img: Input thermal image (numpy array)
        noise_type: Type of noise ('gaussian' or 'salt_pepper')
        severity: Range of noise severity
        
    Returns:
        Noisy thermal image
    """
    # Preserve original dtype
    original_dtype = img.dtype
    
    # Convert to float32 for processing gaussian noise
    if noise_type == 'gaussian' and not np.issubdtype(img.dtype, np.floating):
        img_proc = img.astype(np.float32)
    else:
        img_proc = img.copy()
    
    if noise_type == 'gaussian':
        # Add Gaussian noise
        noise_level = random.uniform(severity[0], severity[1])
        noise = np.random.normal(0, noise_level, img.shape).astype(np.float32)
        noisy_img = img_proc + noise
        
        # Clip values if needed
        if np.issubdtype(original_dtype, np.integer):
            noisy_img = np.clip(noisy_img, 0, 255)
            
        # Convert back to original dtype
        if noisy_img.dtype != original_dtype:
            noisy_img = noisy_img.astype(original_dtype)
            
        return noisy_img
    
    elif noise_type == 'salt_pepper':
        # For salt and pepper, make a copy of the original image
        noisy_img = img_proc.copy()
        
        # Add salt and pepper noise
        noise_severity = random.uniform(severity[0], severity[1]) / 1000
        
        # Salt
        salt_mask = np.random.random(img.shape) < noise_severity/2
        if np.issubdtype(original_dtype, np.integer):
            noisy_img[salt_mask] = 255
        else:
            noisy_img[salt_mask] = 1.0
        
        # Pepper
        pepper_mask = np.random.random(img.shape) < noise_severity/2
        noisy_img[pepper_mask] = 0
        
        return noisy_img
    
    return img

def inject_hotspots(img, num_spots=(1, 5), size_range=(5, 20), intensity_range=(200, 255)):
    """
    Inject random hot spots in the thermal image to simulate heat sources.
    
    Args:
        img: Input thermal image (numpy array)
        num_spots: Range for number of spots to inject (tuple)
        size_range: Range for size of spots (tuple)
        intensity_range: Range for intensity values of spots (tuple)
        
    Returns:
        Thermal image with injected hot spots
    """
     # Preserve original dtype
    original_dtype = img.dtype
    
    # Convert to appropriate type for processing
    if np.issubdtype(img.dtype, np.floating) and img.dtype != np.float32:
        augmented_img = img.copy().astype(np.float32)
    elif not np.issubdtype(img.dtype, np.floating) and img.dtype != np.uint8:
        augmented_img = img.copy().astype(np.uint8)
    else:
        augmented_img = img.copy()
        
    height, width = img.shape[:2]
    
    num = random.randint(num_spots[0], num_spots[1])
    
    for _ in range(num):
        # Random position
        y = random.randint(0, height - 1)
        x = random.randint(0, width - 1)
        
        # Random size
        size = random.randint(size_range[0], size_range[1])
        
        # Random intensity
        intensity = random.randint(intensity_range[0], intensity_range[1])
        
        # Draw the spot
        cv2.circle(augmented_img, (x, y), size, intensity, -1)
    
    # Convert back to original dtype if needed
    if augmented_img.dtype != original_dtype:
        if np.issubdtype(original_dtype, np.integer):
            augmented_img = np.clip(augmented_img, 0, 255)
        augmented_img = augmented_img.astype(original_dtype)
    
    return augmented_img

def inject_coldspots(img, num_spots=(1, 5), size_range=(5, 20), intensity_range=(0, 50)):
    """
    Inject random cold spots in the thermal image to simulate cold regions.
    
    Args:
        img: Input thermal image (numpy array)
        num_spots: Range for number of spots to inject (tuple)
        size_range: Range for size of spots (tuple)
        intensity_range: Range for intensity values of spots (tuple)
        
    Returns:
        Thermal image with injected cold spots
    """
    # Preserve original dtype
    original_dtype = img.dtype
    
    # Convert to appropriate type for processing
    if np.issubdtype(img.dtype, np.floating) and img.dtype != np.float32:
        augmented_img = img.copy().astype(np.float32)
    elif not np.issubdtype(img.dtype, np.floating) and img.dtype != np.uint8:
        augmented_img = img.copy().astype(np.uint8)
    else:
        augmented_img = img.copy()
        
    height, width = img.shape[:2]
    
    num = random.randint(num_spots[0], num_spots[1])
    
    for _ in range(num):
        y = random.randint(0, height - 1)
        x = random.randint(0, width - 1)
        size = random.randint(size_range[0], size_range[1])
        intensity = random.randint(intensity_range[0], intensity_range[1])
        cv2.circle(augmented_img, (x, y), size, intensity, -1)
    
    # Convert back to original dtype if needed
    if augmented_img.dtype != original_dtype:
        if np.issubdtype(original_dtype, np.integer):
            augmented_img = np.clip(augmented_img, 0, 255)
        augmented_img = augmented_img.astype(original_dtype)
    
    return augmented_img

def simulate_emissivity(img, grid_size=(32, 32), variation_range=(0.7, 1.0)):
    """
    Simulate varying emissivity of different surfaces by applying a
    multiplicative grid pattern.
    
    Args:
        img: Input thermal image (numpy array)
        grid_size: Size of the emissivity grid cells (tuple)
        variation_range: Range of emissivity values (tuple)
        
    Returns:
        Thermal image with simulated emissivity variations
    """
    # Preserve original dtype
    original_dtype = img.dtype
    
    # Always work with float32 for emissivity simulation
    augmented_img = img.copy().astype(np.float32)
    
    height, width = img.shape[:2]
    
    # Create a grid of emissivity values
    cell_height, cell_width = grid_size
    rows, cols = (height + cell_height - 1) // cell_height, (width + cell_width - 1) // cell_width
    
    # Generate random emissivity values for each grid cell
    emissivity_grid = np.random.uniform(variation_range[0], variation_range[1], (rows, cols))
    
    # Apply the emissivity grid to the image
    for i in range(rows):
        for j in range(cols):
            r_start, r_end = i * cell_height, min((i+1) * cell_height, height)
            c_start, c_end = j * cell_width, min((j+1) * cell_width, width)
            
            augmented_img[r_start:r_end, c_start:c_end] *= emissivity_grid[i, j]
    
    # Convert back to original dtype
    if original_dtype != np.float32:
        if np.issubdtype(original_dtype, np.integer):
            augmented_img = np.clip(augmented_img, 0, 255)
        augmented_img = augmented_img.astype(original_dtype)
    
    return augmented_img

def add_thermal_reflection(img, reflection_strength=(0.1, 0.3)):
    """
    Add simulated reflections often seen in thermal imaging.
    
    Args:
        img: Input thermal image (numpy array)
        reflection_strength: Range of reflection intensity (tuple)
        
    Returns:
        Thermal image with added reflections
    """
    # Preserve original dtype
    original_dtype = img.dtype
    
    # Convert to float32 for processing
    if img.dtype != np.float32:
        img_float = img.astype(np.float32)
    else:
        img_float = img.copy()
    
    height, width = img_float.shape[:2]
    strength = random.uniform(reflection_strength[0], reflection_strength[1])
    
    # Create a blurred, shifted copy to simulate reflection
    reflection = cv2.GaussianBlur(img_float, (21, 21), 5)
    
    # Apply random geometric transformation to reflection
    angle = random.uniform(-30, 30)
    scale = random.uniform(0.8, 1.2)
    tx = random.randint(-width//10, width//10)
    ty = random.randint(-height//10, height//10)
    
    # Create transformation matrix
    M = cv2.getRotationMatrix2D((width/2, height/2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    
    # Apply transformation
    reflection = cv2.warpAffine(reflection, M, (width, height), borderMode=cv2.BORDER_REFLECT)
    
    # Blend original and reflection
    result = cv2.addWeighted(img_float, 1.0, reflection, strength, 0)
    
    # Convert back to original dtype
    if original_dtype != np.float32:
        if np.issubdtype(original_dtype, np.integer):
            result = np.clip(result, 0, 255)
        result = result.astype(original_dtype)
    
    return result

def thermal_blur(img, kernel_range=(3, 13)):
    """
    Apply specific blur patterns to simulate thermal diffusion effects.
    
    Args:
        img: Input thermal image (numpy array)
        kernel_range: Range for blur kernel size (tuple)
        
    Returns:
        Blurred thermal image
    """
    # Preserve original dtype
    original_dtype = img.dtype
    
    # Choose random kernel size (must be odd)
    kernel_size = random.randrange(kernel_range[0], kernel_range[1], 2)
    
    # Choose blur type
    blur_type = random.choice(['gaussian', 'median', 'bilateral'])
    
    # Check if we need to convert for processing
    needs_uint8 = blur_type == 'median' or (blur_type == 'bilateral' and img.dtype != np.float32)
    
    if needs_uint8:
        # Both medianBlur and bilateralFilter can work with uint8
        if img.dtype != np.uint8:
            # Convert to uint8 for processing
            if np.issubdtype(img.dtype, np.floating):
                # Scale float images to 0-255 range
                img_for_filter = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            else:
                # For other integer types, just clip and convert
                img_for_filter = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img_for_filter = img
    else:
        # For Gaussian blur or when using bilateral with float32
        if blur_type == 'bilateral' and img.dtype != np.float32:
            img_for_filter = img.astype(np.float32)
        else:
            img_for_filter = img
    
    # Apply the selected blur
    if blur_type == 'gaussian':
        blurred_img = cv2.GaussianBlur(img_for_filter, (kernel_size, kernel_size), 0)
    elif blur_type == 'median':
        blurred_img = cv2.medianBlur(img_for_filter, kernel_size)
    elif blur_type == 'bilateral':
        sigma = random.randint(30, 100)
        blurred_img = cv2.bilateralFilter(img_for_filter, kernel_size, sigma, sigma)
    
    # Convert back to original dtype if needed
    if blurred_img.dtype != original_dtype:
        if np.issubdtype(original_dtype, np.floating):
            if needs_uint8:
                # Convert back from uint8 to float, rescaling to original range
                blurred_img = blurred_img.astype(np.float32) / 255.0
        else:
            # For integer types, just clip and convert
            blurred_img = np.clip(blurred_img, 0, 255)
            
        blurred_img = blurred_img.astype(original_dtype)
    
    return blurred_img

def apply_thermal_augmentations(img, augmentations=None):
    """
    Apply a random selection of thermal-specific augmentations.
    
    Args:
        img: Input thermal image (numpy array)
        augmentations: List of augmentations to choose from (if None, uses all)
        
    Returns:
        Augmented thermal image
    """
    # Store original data type for later conversion
    original_dtype = img.dtype
    
    # Convert input to float32 for processing to avoid precision issues
    if not np.issubdtype(img.dtype, np.floating):
        img = img.astype(np.float32)
    
    all_augmentations = [
        temperature_scaling,
        lambda x: add_thermal_noise(x, random.choice(['gaussian', 'salt_pepper'])),
        simulate_emissivity,
        add_thermal_reflection,
        thermal_blur
    ]
    
    if augmentations is None:
        augmentations = all_augmentations
    
    # Apply 1-3 random augmentations
    num_augs = random.randint(1, 3)
    selected_augs = random.sample(augmentations, min(num_augs, len(augmentations)))
    
    result = img.copy()
    for aug_func in selected_augs:
        result = aug_func(result)
    
    # Convert back to original dtype before returning
    if original_dtype != result.dtype:
        if np.issubdtype(original_dtype, np.integer):
            result = np.clip(result, 0, 255)
        result = result.astype(original_dtype)
    
    return result

def random_scale_thermal(img, gt, scales):
    """
    Scale both thermal image and ground truth labels.
    Similar to random_scale in transforms.py but optimized for thermal images.
    
    Args:
        img: Input thermal image (numpy array)
        gt: Ground truth segmentation mask (numpy array)
        scales: List of possible scale factors
        
    Returns:
        Scaled thermal image and ground truth, along with the chosen scale
    """
    scale = random.choice(scales)
    sh = int(img.shape[0] * scale)
    sw = int(img.shape[1] * scale)
    
    # Use INTER_LINEAR for thermal images as it preserves intensity values better
    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    
    # Use INTER_NEAREST for ground truth to preserve label values
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    
    return img, gt, scale

def demo_thermal_augmentations(thermal_img_path, save_results=True, output_dir='./augmented_samples'):
    """
    Demonstrate thermal augmentations by applying them to a sample image.
    
    Args:
        thermal_img_path: Path to a thermal image
        save_results: Whether to save the augmented images
        output_dir: Directory to save the augmented images
        
    Returns:
        Dictionary of augmented images
    """
    import os
    import matplotlib.pyplot as plt
    
    # Create output directory if it doesn't exist
    if save_results and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the thermal image
    thermal_img = cv2.imread(thermal_img_path, cv2.IMREAD_GRAYSCALE)
    if thermal_img is None:
        raise ValueError(f"Could not read image at {thermal_img_path}")
    
    # Apply each augmentation individually
    results = {
        'original': thermal_img,
        'temperature_scaling': temperature_scaling(thermal_img),
        'gaussian_noise': add_thermal_noise(thermal_img, 'gaussian'),
        'salt_pepper_noise': add_thermal_noise(thermal_img, 'salt_pepper'),
        'hotspots': inject_hotspots(thermal_img),
        'coldspots': inject_coldspots(thermal_img),
        'emissivity': simulate_emissivity(thermal_img),
        'reflection': add_thermal_reflection(thermal_img),
        'thermal_blur': thermal_blur(thermal_img),
        'combined': apply_thermal_augmentations(thermal_img)
    }
    
    # Save or display the augmented images
    if save_results:
        for name, img in results.items():
            output_path = os.path.join(output_dir, f"{name}.png")
            cv2.imwrite(output_path, img)
            print(f"Saved {output_path}")
    
    # Visualize the results
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    axs = axs.flatten()
    
    for i, (name, img) in enumerate(results.items()):
        axs[i].imshow(img, cmap='inferno')  # Using inferno colormap for thermal images
        axs[i].set_title(name)
        axs[i].axis('off')
    
    plt.tight_layout()
    if save_results:
        plt.savefig(os.path.join(output_dir, "thermal_augmentations_comparison.png"))
    plt.show()
    
    return results 