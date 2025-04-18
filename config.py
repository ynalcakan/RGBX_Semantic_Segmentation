import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse
from torch.nn.parallel import DistributedDataParallel
import torch

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
C.feature_dim = 512  # Node feature dimension
C.gat_hidden_dim = 512  # GAT hidden dimension
C.gat_num_layers = 2  # Number of GAT layers
C.gat_heads = 4  # Number of attention heads per layer
C.gat_dropout = 0.1  # Dropout rate for GAT
C.use_gatv2 = True  # Use GATv2 instead of GAT
C.graph_fusion_mode = 'weighted'  # Options: 'add', 'weighted', 'concat'

# Hierarchical graph parameters
C.use_hierarchical_graph = True  # Whether to use hierarchical graph structure based on patch pyramid
C.inter_level_edges = True       # Whether to add edges between different levels of the hierarchy

# Cross-attention fusion parameters
C.cross_attn_dim = 64  # Hidden dimension in cross-attention
C.cross_attn_heads = 4  # Number of attention heads in cross-attention
C.cross_attn_window_size = 8  # Window size for efficient attention

# Feature extractor configuration
C.feature_extractor = 'SimpleCNN'  # Options: 'SimpleCNN', 'ResNet', 'MobileNet', 'ViT'
C.fe_pretrained = True  # Whether to use pretrained weights (for supported extractors)
C.fe_freeze_backbone = False  # Whether to freeze backbone weights

# Demo image path (for visualization purposes)
C.demo_image_path = "segmentation.jpg"  # Default fallback image
C.demo_thermal_path = None  # If None, a synthetic thermal image will be created from the RGB image


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
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 2
C.nepochs = 2
C.niters_per_epoch = C.num_train_imgs // C.batch_size  + 1
C.num_workers = 0
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10

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
C.checkpoint_start_epoch = 350
C.checkpoint_step = 50

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