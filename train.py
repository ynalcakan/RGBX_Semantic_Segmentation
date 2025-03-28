import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

from config import config
from dataloader.dataloader import get_train_loader
from models.builder import EncoderDecoder as segmodel
from dataloader.RGBXDataset import RGBXDataset
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR, StepLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor
from utils.loss_opr import FocalLoss2d, RCELoss, BalanceLoss, berHuLoss, SigmoidFocalLoss, TopologyAwareLoss

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
logger = get_logger()

os.environ['MASTER_PORT'] = '169710'

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True
    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader
    train_loader, train_sampler = get_train_loader(engine, RGBXDataset)

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

    # config network and criterion
    FL_gamma = config.FL_gamma
    FL_alpha = config.FL_alpha

    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d
    
    # Initialize criterion with correct background value
    if config.criterion == 'SigmoidFocalLoss':
        criterion = SigmoidFocalLoss(ignore_label=config.background, gamma=config.FL_gamma, alpha=config.FL_alpha, reduction='mean')
    elif config.criterion == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
    elif config.criterion == 'BalanceLoss':
        criterion = BalanceLoss(ignore_index=config.background, reduction='mean')
    elif config.criterion == 'RCELoss':
        criterion = RCELoss(ignore_index=config.background, reduction='mean')
    elif config.criterion == 'berHuLoss':
        criterion = berHuLoss(ignore_index=config.background, reduction='mean')
    elif config.criterion == "FocalLoss2d":
        criterion = FocalLoss2d(ignore_index=config.background, reduction='mean')
    elif config.criterion == 'CE_Focal':
        # multiple loss function
        criterion1 = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
        criterion2 = SigmoidFocalLoss(ignore_label=config.background, gamma=config.FL_gamma, alpha=config.FL_alpha, reduction='mean')
        criterion = (criterion1, criterion2)
    elif config.criterion == 'TopologyAwareCE':
        # Combine CrossEntropy with Topology loss
        criterion1 = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
        criterion2 = TopologyAwareLoss(ignore_index=config.background, reduction='mean')
        criterion = (criterion1, criterion2)
    else:
        raise NotImplementedError

    # Store fixed samples for visualization
    fixed_samples = None
    
    model = segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)
    
    # group weight and config optimizer
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr

        params_list = []
        params_list = group_weight(params_list, model, BatchNorm2d, base_lr)
        
        if config.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
        elif config.optimizer == 'SGDM':
            optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
        elif config.optimizer == 'LBFGS':
            optimizer = torch.optim.LBFGS(params_list, lr=base_lr, max_iter=20, max_eval=None, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100, line_search_fn=None)
        else:
            raise NotImplementedError
    else:
        base_lr = config.lr
        
        params_list = []
        params_list = group_weight(params_list, model, BatchNorm2d, base_lr)
        
        if config.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
        elif config.optimizer == 'SGDM':
            optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
        elif config.optimizer == 'LBFGS':
            optimizer = torch.optim.LBFGS(params_list, lr=base_lr, max_iter=20, max_eval=None, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100, line_search_fn=None)
        else:
            raise NotImplementedError

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    if engine.distributed:
        logger.info('.............distributed training.............')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model, device_ids=[engine.local_rank], 
                                            output_device=engine.local_rank, find_unused_parameters=False)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model,optimizer=optimizer)

    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()
    model.train()
    logger.info('begin trainning:')
    
    for epoch in range(engine.state.epoch, config.nepochs+1):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)

        sum_loss = 0

        for idx in pbar:
            engine.update_iteration(epoch, idx)

            minibatch = next(dataloader)
            imgs = minibatch['data']
            gts = minibatch['label']
            modal_xs = minibatch['modal_x']

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            modal_xs = modal_xs.cuda(non_blocking=True)

            aux_rate = 0.2
            loss = model(imgs, modal_xs, gts)

            # reduce the whole loss over multi-gpu
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = (epoch- 1) * config.niters_per_epoch + idx 
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            if engine.distributed:
                sum_loss += reduce_loss.item()
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                        + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (reduce_loss.item(), (sum_loss / (idx + 1)))
            else:
                sum_loss += loss
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                        + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))

            del loss
            pbar.set_description(print_str, refresh=False)
            
            # Log learning rate and visualize samples
            if ((engine.distributed and (engine.local_rank == 0)) or (not engine.distributed)):
                # Log learning rate every 100 iterations
                if idx % 100 == 0:
                    tb.add_scalar('train/learning_rate', lr, current_idx)
                
                # Visualize sample predictions (once per epoch)
                if idx == 0:
                    with torch.no_grad():
                        # Store fixed samples in the first epoch
                        if fixed_samples is None and epoch == 1:
                            fixed_samples = {
                                'imgs': imgs[2:4].clone(),  # Store 2 samples
                                'gts': gts[2:4].clone(),
                                'modal_xs': modal_xs[2:4].clone()
                            }
                        
                        # Use fixed samples for visualization
                        vis_imgs = fixed_samples['imgs'] if fixed_samples is not None else imgs[2:4]
                        vis_gts = fixed_samples['gts'] if fixed_samples is not None else gts[2:4]
                        vis_modal_xs = fixed_samples['modal_xs'] if fixed_samples is not None else modal_xs[2:4]
                        
                        # Get model predictions
                        if isinstance(model, DistributedDataParallel):
                            pred = model.module(vis_imgs, vis_modal_xs)
                        else:
                            pred = model(vis_imgs, vis_modal_xs)
                        
                        # Convert predictions to visualizable format
                        if isinstance(pred, tuple):  # If model returns multiple outputs
                            pred = pred[0]  # Take the main prediction
                        pred = torch.argmax(pred, dim=1)  # Convert to class indices
                        
                        # Log a few sample images with their predictions
                        for b in range(min(2, vis_imgs.size(0))):  # Log up to 2 samples
                            # Save static images only in the first epoch
                            if epoch == 1:
                                # Original RGB image - normalize to [0,1] range and convert BGR to RGB
                                rgb_img = vis_imgs[b].clone()
                                rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
                                # Convert BGR to RGB by swapping channels
                                rgb_img = rgb_img[[2,1,0], :, :]  # Reorder channels from BGR to RGB
                                tb.add_image(f'sample_{b}/rgb', 
                                           rgb_img, 0)  # Use step 0 for static images
                                
                                # Ground truth visualization with colors
                                colormap = torch.tensor([
                                    [0, 0, 0],        # unlabeled - black
                                    [0, 0, 255],      # fire extinguisher - blue
                                    [0, 255, 0],      # backpack - green
                                    [255, 0, 0],      # hand drill - red
                                    [255, 255, 255],  # rescue randy - white
                                ], device=vis_gts.device).float() / 255.0  # Normalize to [0,1]
                                
                                H, W = vis_gts[b].shape[-2:]
                                gt_vis = torch.zeros((3, H, W), device=vis_gts.device)
                                for i in range(len(colormap)):
                                    mask = (vis_gts[b] == i)
                                    for c in range(3):
                                        gt_vis[c][mask] = colormap[i][c]
                                tb.add_image(f'sample_{b}/ground_truth', 
                                           gt_vis, 0)  # Use step 0 for static images
                                
                                # Modal X input - normalize to [0,1] range
                                modal_x_img = vis_modal_xs[b].clone()
                                modal_x_img = (modal_x_img - modal_x_img.min()) / (modal_x_img.max() - modal_x_img.min())
                                tb.add_image(f'sample_{b}/modal_x',
                                           modal_x_img, 0)  # Use step 0 for static images
                            
                            # Prediction visualization with colors (save for every epoch)
                            H, W = vis_gts[b].shape[-2:]
                            pred_vis = torch.zeros((3, H, W), device=pred.device)
                            for i in range(len(colormap)):
                                mask = (pred[b] == i)
                                for c in range(3):
                                    pred_vis[c][mask] = colormap[i][c]
                            tb.add_image(f'sample_{b}/prediction',
                                       pred_vis, epoch)

            if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
                tb.add_scalar('train/epoch_loss', sum_loss / len(pbar), epoch)

            if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
                if engine.distributed and (engine.local_rank == 0):
                    engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                    config.log_dir,
                                                    config.log_dir_link)
                elif not engine.distributed:
                    engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                    config.log_dir,
                                                    config.log_dir_link)
