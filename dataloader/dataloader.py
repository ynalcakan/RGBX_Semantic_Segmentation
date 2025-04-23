import cv2
import torch
import numpy as np
from torch.utils import data
import random
from config import config
from utils.transforms import generate_random_crop_pos, random_crop_pad_to_shape, normalize

def random_mirror(rgb, gt, modal_x):
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)
        gt = cv2.flip(gt, 1)
        modal_x = cv2.flip(modal_x, 1)

    return rgb, gt, modal_x

def random_scale(rgb, gt, modal_x, scales):
    scale = random.choice(scales)
    sh = int(rgb.shape[0] * scale)
    sw = int(rgb.shape[1] * scale)
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)

    return rgb, gt, modal_x, scale

def random_color_jitter(rgb, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    # Convert to HSV (Hue, Saturation, Value) color space
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Apply random brightness
    brightness_factor = 1.0 + random.uniform(-brightness, brightness)
    hsv[:, :, 2] *= brightness_factor

    # Apply random saturation
    saturation_factor = 1.0 + random.uniform(-saturation, saturation)
    hsv[:, :, 1] *= saturation_factor

    # Apply random hue
    hue_factor = random.uniform(-hue, hue)
    hsv[:, :, 0] += hue_factor * 180  # OpenCV uses 0-180 for hue

    # Clip values to valid range
    hsv = np.clip(hsv, 0, 255)

    # Convert back to BGR color space
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return rgb

def random_gaussian_blur(rgb, kernel_size=(5, 5), sigma=1.0):
    if random.random() >= 0.5:
        rgb = cv2.GaussianBlur(rgb, kernel_size, sigma)
    return rgb

def cutout(rgb, gt, modal_x, mask_size=25, p=0.5):
    if random.random() > p:
        return rgb, gt, modal_x

    h, w = rgb.shape[:2]
    mask_size_half = mask_size // 2

    # Randomly choose the center of the mask
    cx = random.randint(mask_size_half, w - mask_size_half)
    cy = random.randint(mask_size_half, h - mask_size_half)

    # Calculate the coordinates of the mask
    x1 = max(0, cx - mask_size_half)
    y1 = max(0, cy - mask_size_half)
    x2 = min(w, cx + mask_size_half)
    y2 = min(h, cy + mask_size_half)

    # Apply the cutout
    rgb[y1:y2, x1:x2, :] = 0
    gt[y1:y2, x1:x2] = 255  # Assuming 255 is the ignore label
    modal_x[y1:y2, x1:x2, :] = 0

    return rgb, gt, modal_x

class TrainPre(object):
    def __init__(self, norm_mean, norm_std):
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __call__(self, rgb, gt, modal_x):
        # Apply horizontal flipping based on config
        if config.enable_random_mirror:
            rgb, gt, modal_x = random_mirror(rgb, gt, modal_x)
            
        # Apply random scaling
        if config.train_scale_array is not None:
            rgb, gt, modal_x, scale = random_scale(rgb, gt, modal_x, config.train_scale_array)

        # Apply color jittering based on config
        if config.enable_color_jitter:
            rgb = random_color_jitter(rgb)
            
        # Apply gaussian blur based on config
        if config.enable_gaussian_blur:
            rgb = random_gaussian_blur(rgb)
            
        # Apply cutout based on config
        if config.enable_cutout:
            rgb, gt, modal_x = cutout(rgb, gt, modal_x)

        rgb = normalize(rgb, self.norm_mean, self.norm_std)
        modal_x = normalize(modal_x, self.norm_mean, self.norm_std)

        crop_size = (config.image_height, config.image_width)
        
        # Apply random crop based on config
        if hasattr(config, 'enable_random_crop') and config.enable_random_crop:
            crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)
            p_rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
            p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
            p_modal_x, _ = random_crop_pad_to_shape(modal_x, crop_pos, crop_size, 0)
        else:
            # If no random crop, just resize to the target size
            p_rgb = cv2.resize(rgb, (crop_size[1], crop_size[0]), interpolation=cv2.INTER_LINEAR)
            p_gt = cv2.resize(gt, (crop_size[1], crop_size[0]), interpolation=cv2.INTER_NEAREST)
            p_modal_x = cv2.resize(modal_x, (crop_size[1], crop_size[0]), interpolation=cv2.INTER_LINEAR)
        
        p_rgb = p_rgb.transpose(2, 0, 1)
        p_modal_x = p_modal_x.transpose(2, 0, 1)
        
        return p_rgb, p_gt, p_modal_x

class ValPre(object):
    def __call__(self, rgb, gt, modal_x):
        return rgb, gt, modal_x

def get_train_loader(engine, dataset):
    data_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root':config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'class_names': config.class_names,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names}
    train_preprocess = TrainPre(config.norm_mean, config.norm_std)

    train_dataset = dataset(data_setting, "train", train_preprocess, config.batch_size * config.niters_per_epoch)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader, train_sampler