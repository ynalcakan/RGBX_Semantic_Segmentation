import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

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

# Class balancing parameters
C.beta = 0.9999  # For ClassBalancedCELoss
C.ma_momentum = 0.9  # For MABalancedCELoss

"""Image Config"""
C.background = 255
C.image_height = 480
C.image_width = 640
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

""" Settings for network, this would be different for each kind of model"""
C.backbone = 'mit_b2' # Remember change the path below.   # Possibilities: mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5, swin_s, swin_b
C.pretrained_model = C.root_dir + '/pretrained/segformer/mit_b2.pth'
C.decoder = 'UPernet'  # Possibilities: MLPDecoder, UPernet, deeplabv3+, None
C.decoder_embed_dim = 512 # Output dimension that decoder will project features to. Input dimensions are determined by backbone. # 512 b2, 768 b4,
C.rectify_module = 'FRM'  # Possibilities: FRM, IFRM, IFRMv2
C.fusion_module = 'GFM'  # Possibilities: FFM, IFFM, GFM
C.gfm_net_type = 'GCNNetworkV5'  # Possibilities: GCNNetwork, GCNNetworkV2, GCNNetworkV3, GCNNetworkV4, GCNNetworkV5
C.optimizer = 'AdamW'
C.criterion = 'CE_SoftEdgeLoss'    # Possibilities: SigmoidFocalLoss, CrossEntropyLoss, ClassBalancedCELoss, BatchBalancedCELoss, MABalancedCELoss, MedianFreqCELoss, CE_CannyEdgeLoss, CE_SoftEdgeLoss
C.GCN_layers = 2
C.GCN_dropout_rate = 0.1
C.sag_pool_ratio = 0.5

# Integration weights for combined losses
C.loss_weight1 = 1.0    # Weight for primary segmentation loss (CrossEntropy or other)      # default 1.0
C.loss_weight2 = 0.5  # Weight for edge loss                                                # default 0.5

# SigmoidFocalLoss parameters
C.FL_gamma = 4.0     
C.FL_alpha = 0.25

"""LR Config"""
C.lr_method = 'CosineAnnealingWarmupLR' # 'OneCycleLR', 'WarmUpPolyLR', 'StepLR', 'CosineAnnealingWarmupLR', 'ReduceLROnPlateauLR', 'CyclicLR', 'MultiStageLR', 'LinearIncreaseLR'
C.lr = 1e-4  # Slightly higher than current 3e-5 for better exploration # try 3e-1 for more epochs
C.lr_power = 0.8 # 0.9  
C.warm_up_epoch = 25  # ~5-6% of total epochs (450) is optimal for warmup
C.min_lr = 5e-6  # 5% of max learning rate prevents too small gradients 5e-6 pri lr 1e-4
C.factor = 0.1
C.patience = 10
C.threshold = 1e-4
C.cooldown = 0
C.cycle_epochs = 1
C.warmup_epochs = 1
C.weight_decay = 0.01  # Reduce slightly from 0.015 for cosine scheduler

"""Train Config"""
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 12
C.nepochs = 750
C.niters_per_epoch = C.num_train_imgs // C.batch_size  + 1
C.num_workers = 16
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]

# Data augmentation options
C.enable_random_mirror = True         # Enable/disable horizontal flipping during training
C.enable_random_crop = False           # Enable/disable random crop during training
C.enable_color_jitter = False          # Enable/disable color jittering
C.enable_gaussian_blur = True         # Enable/disable Gaussian blur
C.enable_cutout = False                # Enable/disable cutout augmentation

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1


"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1] # [0.75, 1, 1.25] # 
C.eval_flip = False # True # 
C.eval_crop_size = [480, 640] # [height weight]

"""Store Config"""
C.checkpoint_start_epoch = 50
C.checkpoint_step = 50

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir))

if C.criterion == 'SigmoidFocalLoss':
    log_path = 'logs/' + C.dataset_name + '/' + 'log_' + C.backbone + '_' + C.decoder + '_' + C.rectify_module + '_' + C.fusion_module + '_' + C.criterion + '_gamma' + str(C.FL_gamma) + '_alpha' + str(C.FL_alpha)
elif C.fusion_module == 'GFM':
    log_path = 'logs/' + C.dataset_name + '/' + 'log_' + C.backbone + '_' + C.decoder + '_' + C.rectify_module + '_' + C.fusion_module + '_' + C.criterion + '_' + C.gfm_net_type + '_' + 'meangp'
else:
    log_path = 'logs/' + C.dataset_name + '/' + 'log_' + C.backbone + '_' + C.decoder + '_' + C.rectify_module + '_' + C.fusion_module + '_' + C.criterion

C.log_dir = osp.abspath(log_path)
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

if __name__ == '__main__':
    print(config.nepochs)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()

    if args.tensorboard:
        open_tensorboard()