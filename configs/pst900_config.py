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

C.seed = 42

remoteip = os.popen('pwd').read()
C.root_dir = os.path.abspath(os.path.join(os.getcwd(), './'))
C.abs_dir = osp.realpath(".")

# Dataset config
"""Dataset Path"""
C.dataset_name = 'PST900'
C.dataset_path = osp.join(C.root_dir, 'datasets', 'PST900')
C.rgb_root_folder = osp.join(C.dataset_path, 'RGB_resized')
C.rgb_format = '.png'
C.gt_root_folder = osp.join(C.dataset_path, 'Label_resized')
C.gt_format = '.png'
C.gt_transform = False  # Set to False because label 0 is valid in PST900 (it's the background class)
# True when label 0 is invalid, you can also modify the function _transform_gt in dataloader.RGBXDataset
# True for most dataset valid, False for MFNet(?)
C.x_root_folder = osp.join(C.dataset_path, 'Thermal_resized')
C.x_format = '.png'
C.x_is_single_channel = True # True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = False
C.num_train_imgs = 597   # 597 2388
C.num_eval_imgs = 288   # 288 1152
C.num_classes = 5  # background (0) + 4 objects (1-4)
C.class_names = ["unlabeled", "fire extinguisher", "backpack", "hand drill", "rescue randy"]

"""Image Config"""
C.background = 255 
C.image_height = 480
C.image_width = 640
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

""" Settings for network, this would be different for each kind of model"""
C.backbone = 'mit_b2_w_aspp' # Remember change the path below.   # Possibilities: mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5, swin_s, swin_b
C.pretrained_model = C.root_dir + '/pretrained/segformer/mit_b2.pth'
C.decoder = 'UPernet'  # Possibilities: MLPDecoder, UPernet, deeplabv3+, MLPDecoderpp, None
C.decoder_embed_dim = 512
C.optimizer = 'AdamW'
C.criterion = 'CrossEntropyLoss'    # Possibilities: CrossEntropyLoss, FocalLoss, DiceCELoss

# FocalLoss parameters
C.FL_gamma = 4.0     
C.FL_alpha = 0.25

"""Train Config"""
C.lr = 2e-4
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 5e-2   # use smaller if data is small
C.batch_size = 8
C.nepochs = 300
C.niters_per_epoch = C.num_train_imgs // C.batch_size  + 1
C.num_workers = 16
C.train_scale_array = None #[0.5, 0.75, 1, 1.25, 1.5, 1.75]    # augmentation scale
C.warm_up_epoch = 10         # lr warm up epoch

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

if C.criterion == 'FocalLoss':
    log_path = 'logs/' + C.dataset_name + '/' + 'log_' + C.backbone + '_' + C.decoder + '_FRM' + '_' + C.criterion + '_gamma' + str(C.FL_gamma) + '_alpha' + str(C.FL_alpha)
else:
    log_path = 'logs/' + C.dataset_name + '/' + 'log_' + C.backbone + '_' + C.decoder + '_FRM' + '_' + C.criterion 

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