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
from utils.lr_policy import WarmUpPolyLR, StepLR, OneCycleLR, ReduceLROnPlateauLR, CyclicLR, CosineAnnealingWarmupLR, MultiStageLR, LinearIncreaseLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor
from utils.loss_opr import FocalLoss2d, RCELoss, BalanceLoss, berHuLoss, SigmoidFocalLoss, TopologyAwareLoss, ClassBalancedCELoss, BatchBalancedCELoss, MABalancedCELoss, MedianFreqCELoss, DiceLoss, FocalDiceLoss, SoftEdgeLoss

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

    criterion = config.criterion
    if criterion == 'SigmoidFocalLoss':
        criterion = SigmoidFocalLoss(ignore_label=config.background, gamma=FL_gamma, alpha=FL_alpha, reduction='mean')
    elif criterion == 'ClassBalancedCELoss':
        if hasattr(config, 'samples_per_cls'):
            criterion = ClassBalancedCELoss(samples_per_cls=config.samples_per_cls, beta=config.beta if hasattr(config, 'beta') else 0.9999, ignore_index=config.background, reduction='mean')
        else:
            logger.warning("samples_per_cls not found in config, falling back to BatchBalancedCELoss")
            criterion = BatchBalancedCELoss(num_classes=config.num_classes, ignore_index=config.background, reduction='mean')
    elif criterion == 'BatchBalancedCELoss':
        criterion = BatchBalancedCELoss(num_classes=config.num_classes, ignore_index=config.background, reduction='mean')
    elif criterion == 'MABalancedCELoss':
        criterion = MABalancedCELoss(num_classes=config.num_classes, ignore_index=config.background, momentum=config.ma_momentum if hasattr(config, 'ma_momentum') else 0.9)
    elif criterion == 'MedianFreqCELoss':
        criterion = MedianFreqCELoss(num_classes=config.num_classes, ignore_index=config.background)
    elif criterion == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
    elif criterion == 'BalanceLoss':
        criterion = BalanceLoss(ignore_index=config.background, reduction='mean')
    elif criterion == 'RCELoss':
        criterion = RCELoss(ignore_index=config.background, reduction='mean')
    elif criterion == 'berHuLoss':
        criterion = berHuLoss(ignore_index=config.background, reduction='mean')
    elif criterion == "FocalLoss2d":
        criterion = FocalLoss2d(ignore_index=config.background, reduction='mean')
    elif criterion == 'FocalDiceLoss':
        criterion = FocalDiceLoss(ignore_index=config.background)
    elif criterion == 'DiceLoss':
        criterion = DiceLoss(ignore_index=config.background)
    elif criterion == 'CE_Focal':
        # multiple loss function
        criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
        criterion2 = SigmoidFocalLoss(ignore_label=config.background, gamma=FL_gamma, alpha=FL_alpha, reduction='mean')
        criterion = (criterion, criterion2)
    elif criterion == 'MedianFreqCE_Focal':
        # multiple loss function
        criterion = MedianFreqCELoss(num_classes=config.num_classes, ignore_index=config.background)
        criterion2 = SigmoidFocalLoss(ignore_label=config.background, gamma=FL_gamma, alpha=FL_alpha, reduction='mean')
        criterion = (criterion, criterion2)
    elif criterion == 'TopologyAwareCE':
        # Combine CrossEntropy with Topology loss
        criterion1 = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
        criterion2 = TopologyAwareLoss(ignore_index=config.background, reduction='mean')
        criterion = (criterion1, criterion2)
    elif criterion == 'SoftEdgeLoss':
        criterion = SoftEdgeLoss(ignore_index=config.background, reduction='mean')
    elif criterion == 'CE_SoftEdgeLoss':
        criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
        criterion2 = SoftEdgeLoss(ignore_index=config.background, reduction='mean')
        criterion = (criterion, criterion2)
    else:
        raise NotImplementedError


    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d
    
    model=segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)
    
    # group weight and config optimizer
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr

        params_list = []
        params_list = group_weight(params_list, model, BatchNorm2d, base_lr)
        
        if config.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.99), weight_decay=config.weight_decay)
        elif config.optimizer == 'SGDM':
            optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
        elif config.optimizer == 'LBFGS':
            optimizer = torch.optim.LBFGS(params_list, lr=base_lr, max_iter=20, max_eval=None, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100, line_search_fn=None)
        else:
            raise NotImplementedError

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    if config.lr_method == 'WarmUpPolyLR':
        lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)
    elif config.lr_method == 'OneCycleLR':
        # Using OneCycleLR which often works better for segmentation tasks
        lr_policy = OneCycleLR(start_lr=base_lr, max_lr=base_lr*4, total_iters=total_iteration, pct_start=0.3)
    elif config.lr_method == 'StepLR':
        lr_policy = StepLR(base_lr, config.step_size, config.gamma)
    elif config.lr_method == 'CosineAnnealingWarmupLR':
        lr_policy = CosineAnnealingWarmupLR(base_lr, total_iteration, config.warm_up_epoch, config.min_lr)
    elif config.lr_method == 'ReduceLROnPlateauLR':
        lr_policy = ReduceLROnPlateauLR(base_lr, config.factor, config.patience, config.min_lr, config.threshold, config.cooldown)
    elif config.lr_method == 'CyclicLR':
        lr_policy = CyclicLR(base_lr, config.max_lr, config.cycle_epochs, config.warmup_epochs, total_iteration, config.niters_per_epoch)
    elif config.lr_method == 'MultiStageLR':
        lr_policy = MultiStageLR(config.lr_stages)
    elif config.lr_method == 'LinearIncreaseLR':
        lr_policy = LinearIncreaseLR(base_lr, config.end_lr, config.warm_iters)
    else:
        raise NotImplementedError

    if engine.distributed:
        logger.info('.............distributed training.............')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model, device_ids=[engine.local_rank], 
                                            output_device=engine.local_rank, find_unused_parameters=True)
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

            if isinstance(criterion, tuple):
                loss, loss_components = model(imgs, modal_xs, gts)
            else:
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
                
                # Add individual loss components only for multiple losses
                if isinstance(criterion, tuple):
                    # Initialize loss_sums dict if it doesn't exist
                    if not hasattr(engine, 'loss_sums'):
                        engine.loss_sums = {key: 0.0 for key in loss_components.keys()}
                    
                    for key in loss_components.keys():
                        reduced_val = all_reduce_tensor(loss_components[key], world_size=engine.world_size)
                        engine.loss_sums[key] += reduced_val.item()
                        print_str += ' %s=%.4f' % (key, engine.loss_sums[key]/(idx+1))
            else:
                sum_loss += loss
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                        + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))
                
                # Add individual loss components only for multiple losses (non-distributed case)
                if isinstance(criterion, tuple):
                    # Initialize loss_sums dict if it doesn't exist
                    if not hasattr(engine, 'loss_sums'):
                        engine.loss_sums = {key: 0.0 for key in loss_components.keys()}
                    
                    for key in loss_components.keys():
                        engine.loss_sums[key] += loss_components[key].item()
                        print_str += ' %s=%.4f' % (key, engine.loss_sums[key]/(idx+1))

            del loss
            pbar.set_description(print_str, refresh=False)
        
        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)
            
            # Add individual loss components to TensorBoard if using multiple losses
            if isinstance(criterion, tuple) and hasattr(engine, 'loss_sums'):
                for key in engine.loss_sums.keys():
                    tb.add_scalar(f'train_{key}', engine.loss_sums[key] / len(pbar), epoch)
                # Reset loss sums for next epoch
                engine.loss_sums = {key: 0.0 for key in engine.loss_sums.keys()}

        if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)
