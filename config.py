import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse
from torch.nn.parallel import DistributedDataParallel
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import functional as F

C = edict()
config = C
cfg = C

C.seed = 12345

remoteip = os.popen('pwd').read()
C.root_dir = os.path.abspath(os.path.join(os.getcwd(), './'))
C.abs_dir = osp.realpath(".")

# Dataset config
"""Dataset Path"""
C.dataset_name = 'MFNet'
C.dataset_path = osp.join(C.root_dir, '../Datasets', 'MFNet')
C.rgb_root_folder = osp.join(C.dataset_path, 'RGB')
C.rgb_format = '.png'
C.gt_root_folder = osp.join(C.dataset_path, 'Label')
C.gt_format = '.png'
C.gt_transform = False
# True when label 0 is invalid, you can also modify the function _transform_gt in dataloader.RGBXDataset
# True for most dataset valid, False for MFNet(?)
C.x_root_folder = osp.join(C.dataset_path, 'Thermal')
C.x_format = '.png'
C.x_is_single_channel = True # True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input
C.train_source = osp.join(C.dataset_path, "train_val.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = False
C.num_train_imgs = 1176
C.num_eval_imgs = 393
C.num_classes = 9
C.class_names =  ["Unlabeled", "Car", "Person", "Bike", "Curve", "Car Stop", "Guardrail", "Color Cone", "Bump"]

# Graph model configurations
C.use_graph = True # Enable graph creation
C.use_gatv2 = True  # Use GATv2 instead of GAT
C.feature_dim = 320       # Match level 2 native dimension
C.gat_hidden_dim = 384    # Hidden dimension used in GAT layers
C.graph_num_layers = 2  # Number of graph attention layers
C.graph_heads = 4  # Number of attention heads
C.graph_dropout = 0.15  # Dropout rate for graph networks
C.graph_fusion_mode = 'concat'  # Options: 'weighted', 'add', 'concat'
C.fusion_weight = 0.7  # Weight for backbone features in weighted fusion (0-1)
C.graph_feature_level = 2  # Level of features to extract from backbone (0-3)


"""Image Config"""
C.background = 255
C.image_height = 480
C.image_width = 640
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

""" Settings for network, this would be different for each kind of model"""
C.backbone = 'mit_b2' # Remember change the path below.   # Possibilities: mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5, swin_s, swin_b
C.pretrained_model = C.root_dir + '/pretrained/segformer/mit_b2.pth'
C.decoder = 'MLPDecoder'  # Possibilities: MLPDecoder, UPernet, deeplabv3+, None
C.decoder_embed_dim = 512
C.optimizer = 'AdamW'
C.criterion = 'CrossEntropyLoss'    # Possibilities: SigmoidFocalLoss, CrossEntropyLoss

# SigmoidFocalLoss parameters
C.FL_gamma = 4.0     
C.FL_alpha = 0.25

"""Train Config"""
C.lr = 1e-4
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.005       # Reduced from 0.01
C.batch_size = 8 
C.nepochs = 25            # Enough epochs for convergence
C.niters_per_epoch = C.num_train_imgs // C.batch_size  + 1
C.num_workers = 0
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

# GAT training config
C.gat_weight_decay = 0.015   # Keep as is

""""Augmentation"""
# Class weighting for better small object detection
C.use_class_weights = True
# Approximate class frequency based on typical segmentation datasets
# You can adjust these based on your specific dataset statistics
C.class_weights = [0.8, 0.9, 1.2, 1.5, 1.2, 1.5, 1.0, 1.8, 1.5]  # Higher weight for smaller/rare classes

# Add to config.py
C.use_onecycle = True
C.max_lr = 3e-4

# Add data augmentation settings
C.color_jitter = 0.4
C.random_scale_range = (0.5, 2.0)
C.random_crop_size = [384, 512]  # Smaller than full resolution
C.mixup_alpha = 0.2  # Optional - mix multiple images

# Add layer freezing option
C.freeze_backbone_layers = 1  # Freeze first layer of backbone

"""Eval Config"""
C.eval_iter = 20
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1] # [0.75, 1, 1.25] # 
C.eval_flip = False # True # 
C.eval_crop_size = [480, 640] # [height weight]
C.patience = 10           # Stop if no improvement for 10 epochs
C.early_stop_tolerance = 0.005  # Allow 0.5% fluctuation in validation metric without resetting patience
C.eval_interval = 1       # Validate every epoch

"""Store Config"""
C.checkpoint_start_epoch = 15
C.checkpoint_step = 1

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir))

if C.criterion == 'SigmoidFocalLoss':
    log_path = 'logs/' + C.dataset_name + '/' + 'log_' + C.backbone + '_' + C.decoder + '_IFRM' + '_' + C.criterion + '_gamma' + str(C.FL_gamma) + '_alpha' + str(C.FL_alpha)
else:
    log_path = 'logs/' + C.dataset_name + '/' + 'log_' + C.backbone + '_' + C.decoder + '_IFRM' + '_' + C.criterion 

C.log_dir = osp.abspath(log_path)
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

def configure_optimizer(model, lr=6e-5, weight_decay=0.01, gat_weight_decay=0.015):
    # Separate GAT parameters from other parameters
    gat_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'gat' in name.lower() or 'graph_processor' in name.lower():
            gat_params.append(param)
        else:
            other_params.append(param)
    
    # Set up parameter groups with different weight decay values
    param_groups = [
        {'params': other_params, 'weight_decay': weight_decay},
        {'params': gat_params, 'weight_decay': gat_weight_decay}  # Higher weight decay for GAT
    ]
    
    # Create optimizer with parameter groups
    optimizer = torch.optim.AdamW(param_groups, lr=lr)
    
    return optimizer



if __name__ == '__main__':
    print(config.nepochs)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()

    # if args.tensorboard:
    #     open_tensorboard()