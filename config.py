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

# Graph creation parameters
C.create_graph = True  # Enable graph creation
C.feature_dim = 320       # Match level 2 native dimension
C.gat_hidden_dim = 320    # Match feature_dim for simplicity
C.gat_num_layers = 2      # Reduce from 3 to 2
C.gat_heads = 4  # Reduced from 8 to 4 for memory efficiency with level 2 features
C.gat_dropout = 0.15  # Dropout rate for GAT
C.use_gatv2 = True  # Use GATv2 instead of GAT
C.graph_fusion_mode = 'concat'  # Options: 'add', 'weighted', 'concat'
C.graph_feature_level = 2  # Feature level to use for graph: 0 (finest) to 3 (coarsest)

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
# e.g. inverse‑frequency or median‑frequency weights
C.criterion = 'WeightedCrossEntropy2d'    # Possibilities: SigmoidFocalLoss, CrossEntropyLoss, FocalLoss2d, BalanceLoss, MedianFreqCELoss, WeightedCrossEntropy2d

# # inverse‑frequency
# counts = np.array()
# inv_freq = 1.0 / counts
# class_weights = inv_freq / inv_freq.sum() * len(inv_freq)

"""Loss function Config"""
# WeightedCrossEntropy2d parameters
C.class_weights = [0.6, 0.9, 1.0, 1.4, 1.2, 1.5, 1.7, 1.4, 1.2] 

# SigmoidFocalLoss parameters
C.FL_gamma = 4.0     
C.FL_alpha = 0.25

"""Train Config"""
C.use_onecycle = True
C.max_lr = 3e-4
C.lr = 1e-4
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.005       # Reduced from 0.01
C.batch_size = 4               # Reduced from 8 to 4 due to larger graph size from level 2 features
C.nepochs = 60            # Enough epochs for convergence
C.niters_per_epoch = C.num_train_imgs // C.batch_size  + 1
C.num_workers = 0
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

# GAT training config
C.gat_weight_decay = 0.015   # Keep as is

# First 10 epochs: freeze more layers
# Next 10 epochs: freeze fewer layers
# Remaining epochs: train all layers
# Freeze backbone layers
C.freeze_backbone_layers = 1  # Freeze first layer of backbone

# Add these to config.py
C.color_jitter = 0.4
C.random_scale_range = (0.5, 2.0)
C.random_crop_size = [384, 512]  # Smaller than full resolution
C.mixup_alpha = 0.2  # Optional - mix multiple images

"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1] # [0.75, 1, 1.25] # 
C.eval_flip = False # True # 
C.eval_crop_size = [480, 640] # [height weight]
C.patience = 10           # Stop if no improvement for 10 epochs
C.eval_interval = 1       # Validate every epoch

"""Store Config"""
C.checkpoint_start_epoch = 20
C.checkpoint_step = 5

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

    if args.tensorboard:
        open_tensorboard()

    if engine.distributed:
        logger.info('.............distributed training.............')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model, device_ids=[engine.local_rank], 
                                           output_device=engine.local_rank, find_unused_parameters=False)

def __init__(self, segformer_model, in_channels, hidden_channels, out_channels, 
             num_layers=2, heads=4, dropout=0.1, use_gatv2=True):
    super().__init__()
    
    # Feature extractor from the segformer
    self.feature_extractor = SegformerFeatureExtractor(segformer_model)
    
    # Get the graph feature level
    self.feature_level = getattr(segformer_model.cfg, 'graph_feature_level', 3)
    
    # Get input channels based on feature level if different from provided
    if self.feature_level != 3:
        # Segformer MiT-B2 has these channel counts at different levels
        level_channels = [64, 128, 320, 512]  # Default for MiT-B2
        if hasattr(segformer_model, 'channels'):
            level_channels = segformer_model.channels
        in_channels = level_channels[self.feature_level]
    
    # Modality fusion layer for combining RGB and X features
    self.modality_fusion = nn.Sequential(
        nn.Linear(in_channels * 2, hidden_channels),
        nn.LayerNorm(hidden_channels),
        nn.ReLU(),
        nn.Dropout(dropout)
    )