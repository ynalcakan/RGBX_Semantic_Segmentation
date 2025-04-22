import os
import cv2
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from config import config
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from utils.weighted_metric import compute_weighted_score, print_weighted_iou
from dataloader.RGBXDataset import RGBXDataset
from models.builder import EncoderDecoder as segmodel
from dataloader.dataloader import ValPre
from utils.transforms import normalize
from utils.pyt_utils import load_restore_model as load_model

logger = get_logger()

class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        modal_x = data['modal_x']
        name = data['fn']
        
        # No need to extract graph_data - it's handled internally by the model
        
        # Call the sliding eval function without graph data parameter
        pred = self.sliding_eval_rgbX(img, modal_x, config.eval_crop_size, 
                                      config.eval_stride_rate, device)
        
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

        if self.save_path is not None:
            ensure_dir(self.save_path)
            ensure_dir(self.save_path+'_color')

            fn = name + '.png'

            # save colored result
            result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
            class_colors = self.dataset.get_class_colors()
            palette_list = list(np.array(class_colors).flat)
            if len(palette_list) < 768:
                palette_list += [0] * (768 - len(palette_list))
            result_img.putpalette(palette_list)
            result_img.save(os.path.join(self.save_path+'_color', fn))

            # save raw result
            cv2.imwrite(os.path.join(self.save_path, fn), pred)
            logger.info('Save the image ' + fn)

        if self.show_image:
            colors = self.dataset.get_class_colors
            image = img
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean,
                                label,
                                pred)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        return results_dict
    
    def pad_image_to_shape(self, img, crop_size, stride=None):
        """
        Pad image to make dimensions divisible by crop_size
        Args:
            img: input image
            crop_size: tuple or int
            stride: tuple or int (not used here)
        Returns:
            padded_img: padded image
            margin: padding margins (top, bottom, left, right)
        """
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
            
        img_height, img_width = img.shape[0:2]
        pad_height = max(crop_size[0] - img_height, 0)
        pad_width = max(crop_size[1] - img_width, 0)
        
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        
        margin = (pad_top, pad_bottom, pad_left, pad_right)
        
        if pad_height > 0 or pad_width > 0:
            padded_img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, 
                                            cv2.BORDER_CONSTANT, value=0)
        else:
            padded_img = img
            
        return padded_img, margin
    
    def sliding_eval_rgbX(self, img, modal_x, crop_size, stride_rate, device):
        """
        Sliding evaluation with support for graph data
        """
        ori_rows, ori_cols = img.shape[:-1]
        processed_pred = np.zeros((ori_rows, ori_cols))

        for s in self.multi_scales:
            img_scale = cv2.resize(img, None, fx=s, fy=s,
                                   interpolation=cv2.INTER_LINEAR)
            modal_x_scale = cv2.resize(modal_x, None, fx=s, fy=s,
                                   interpolation=cv2.INTER_LINEAR)

            new_rows, new_cols = img_scale.shape[:-1]
            processed_pred += self.scale_process_rgbX(img_scale, modal_x_scale,
                                                (ori_rows, ori_cols),
                                                crop_size, stride_rate, device)

        pred = processed_pred.astype(np.uint8)

        return pred

    def scale_process_rgbX(self, img, modal_x, ori_shape, crop_size, stride_rate, device=None):
        """
        Process scaled images with graph data
        """
        new_rows, new_cols = img.shape[:-1]
        long_size = new_cols if new_cols > new_rows else new_rows

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        if long_size <= min(crop_size[0], crop_size[1]):
            # Use original size for inference when the image is small
            input_data, input_modal_x, margin = self.process_image_rgbX(img, modal_x, crop_size)
            score = self.net_process(input_data, input_modal_x)
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]
        else:
            # Use sliding window for large images
            stride = (int(np.ceil(crop_size[0] * stride_rate)),
                      int(np.ceil(crop_size[1] * stride_rate)))
            img_pad, margin = self.pad_image_to_shape(img, crop_size)
            modal_x_pad, _ = self.pad_image_to_shape(modal_x, crop_size)

            score = self.overlay_picture_rgbX(img_pad, modal_x_pad, crop_size, stride)
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]

        score = score.permute(1, 2, 0)
        processed = cv2.resize(score.cpu().numpy(), (ori_shape[1], ori_shape[0]),
                               interpolation=cv2.INTER_LINEAR)
        pred = np.argmax(processed, axis=2)

        return pred

    def net_process(self, inputs, modal_x):
        """
        Process one image with graph data
        """
        # Use the same device as inputs
        device = inputs.device
        
        # Make sure model is on the correct device
        self.val_func = self.val_func.to(device)
        
        # No need to move inputs since we already did in overlay_picture_rgbX
        
        # Use the val_func to get predictions
        with torch.no_grad():
            outputs = self.val_func(inputs, modal_x)
            
        return outputs

    def overlay_picture_rgbX(self, img, modal_x, crop_size, stride):
        """
        Overlay sliding windows for prediction, supporting graph data
        """
        # Get the device from the first input tensor to ensure consistency
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            
        # Determine sliding window locations
        windows = self.compute_sliding_windows(img, crop_size, stride)
        
        # Prepare counting array for averaging overlapping areas
        count = torch.zeros(1, img.shape[0], img.shape[1], device=device)
        score = torch.zeros(config.num_classes, img.shape[0], img.shape[1], device=device)
        
        for window in windows:
            # Extract window coordinates
            x1, y1, x2, y2 = window
            
            # Extract image region
            img_window = img[y1:y2, x1:x2]
            modal_x_window = modal_x[y1:y2, x1:x2]
            
            # Create tensor from numpy array
            input_data, input_modal_x, margin = self.process_image_rgbX(img_window, modal_x_window, crop_size)
            
            # Ensure inputs are on the correct device
            input_data = input_data.to(device)
            input_modal_x = input_modal_x.to(device)
            
            # Run inference
            output = self.net_process(input_data, input_modal_x)
            
            # If output has 4 dimensions [1, C, H, W], remove the batch dimension
            if output.dim() == 4:
                output = output.squeeze(0)
                
            # Ensure output is on the right device
            output = output.to(device)
                
            # Remove padding
            output = output[:, margin[0]:(output.shape[1] - margin[1]),
                      margin[2]:(output.shape[2] - margin[3])]
            
            # Add to score and count
            score[:, y1:y2, x1:x2] += output
            count[:, y1:y2, x1:x2] += 1
        
        # Average the overlapping areas
        score = score / count
        return score
    
    def compute_sliding_windows(self, img, crop_size, stride):
        """Compute all sliding window coordinates for an image"""
        h, w = img.shape[:2]
        x_steps = max(1, int((w - crop_size[1]) / stride[1]) + 1)
        y_steps = max(1, int((h - crop_size[0]) / stride[0]) + 1)
        
        windows = []
        for y in range(y_steps):
            for x in range(x_steps):
                # Calculate coordinates
                x1 = x * stride[1]
                y1 = y * stride[0]
                # Handle edge cases
                x2 = min(x1 + crop_size[1], w)
                y2 = min(y1 + crop_size[0], h)
                # Adjust x1, y1 if needed to maintain crop size
                x1 = max(0, x2 - crop_size[1])
                y1 = max(0, y2 - crop_size[0])
                
                windows.append((x1, y1, x2, y2))
        
        return windows
    
    def process_image_rgbX(self, img, modal_x, crop_size=None):
        """Process image before evaluation"""
        p_img = img
        p_modal = modal_x
        
        if crop_size is not None:
            p_img, margin = self.pad_image_to_shape(p_img, crop_size)
            p_modal, _ = self.pad_image_to_shape(p_modal, crop_size)
            
        p_img = p_img.transpose(2, 0, 1)
        p_modal = p_modal.transpose(2, 0, 1)
        
        # Reshape norm_mean and norm_std for broadcasting
        norm_mean = self.norm_mean.reshape(3, 1, 1)
        norm_std = self.norm_std.reshape(3, 1, 1)
        
        # Normalize images
        p_img = normalize(p_img, norm_mean, norm_std)
        p_modal = normalize(p_modal, norm_mean, norm_std)
        
        p_img = np.expand_dims(p_img, axis=0)
        p_modal = np.expand_dims(p_modal, axis=0)
        
        p_img = torch.from_numpy(p_img).float()
        p_modal = torch.from_numpy(p_modal).float()
        
        return p_img, p_modal, margin

    def compute_metric(self, results):
        hist = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        # First, get standard metrics for comparison
        iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
        
        # Get class weights from config
        class_weights = getattr(config, 'class_weights', None)
        
        # If class weights exist, compute weighted metrics
        if class_weights is not None:
            # Compute weighted metrics
            iou, weighted_iou, mean_IoU, weighted_mean_IoU, freq_IoU, mean_pixel_acc, weighted_mean_pixel_acc, pixel_acc = compute_weighted_score(
                hist, correct, labeled, class_weights
            )
            
            # Print both standard and weighted metrics
            result_line = print_weighted_iou(
                iou, weighted_iou, mean_IoU, weighted_mean_IoU,
                freq_IoU, mean_pixel_acc, weighted_mean_pixel_acc,
                pixel_acc, dataset.class_names, no_print=False
            )
        else:
            # If no weights provided, use standard metrics
            result_line = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
                                dataset.class_names, show_no_back=False)
        
        return result_line

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = segmodel(cfg=config, criterion=config.criterion, norm_layer=nn.BatchNorm2d)
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
    val_pre = ValPre()
    dataset = RGBXDataset(data_setting, 'val', val_pre)
 
    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.norm_mean,
                                 config.norm_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, args.save_path,
                                 args.show_image)
        segmentor.run(config.checkpoint_dir, args.epochs, config.val_log_file,
                      config.link_val_log_file)