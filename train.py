import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm
import pathlib

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

# Helper function to synchronize flag detection across processes
def check_stop_flag(engine):
    """Check for stop flag and broadcast result to all processes"""
    flag_exists = torch.tensor([0], device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Only rank 0 checks for the flag file
    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        stop_flag = pathlib.Path("stop_training.flag")
        if stop_flag.exists():
            flag_exists[0] = 1
    
    # Broadcast result to all processes
    if engine.distributed:
        torch.distributed.broadcast(flag_exists, 0)
        
    return flag_exists[0] == 1

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
    
    else:
        tb = None # Ensure tb is defined even if not initialized

    # config network and criterion
    FL_gamma = config.FL_gamma
    FL_alpha = config.FL_alpha

    criterion = config.criterion
    if criterion == 'SigmoidFocalLoss':
        criterion = SigmoidFocalLoss(ignore_label=config.background, gamma=FL_gamma, alpha=FL_alpha, reduction='mean')
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
    elif criterion == 'CE_Focal':
        # multiple loss function
        criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
        criterion2 = SigmoidFocalLoss(ignore_label=config.background, gamma=FL_gamma, alpha=FL_alpha, reduction='mean')
        criterion = (criterion, criterion2)
    elif criterion == 'TopologyAwareCE':
        # Combine CrossEntropy with Topology loss
        criterion1 = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
        criterion2 = TopologyAwareLoss(ignore_index=config.background, reduction='mean')
        criterion = (criterion1, criterion2)
    else:
        raise NotImplementedError

    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d
    
    
    model=segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)

    # group weight and config optimizer
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
            # Set static graph to fix gradient checkpointing issues
            model._set_static_graph()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model,optimizer=optimizer)

    if engine.continue_state_object:
        engine.restore_checkpoint()
    else:
        optimizer.zero_grad()
        model.train()
        logger.info('begin trainning:')
    
    # Initialize early stopping variables
    best_loss = float('inf')
    patience_counter = 0
    tolerance = config.early_stop_tolerance if hasattr(config, 'early_stop_tolerance') else 0.0

    for epoch in range(engine.state.epoch, config.nepochs+1):
        logger.info(f"--> [Epoch {epoch}] Starting...")
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
            
            # No need to handle external graph_data as it's now integrated in the model
            
            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            modal_xs = modal_xs.cuda(non_blocking=True)

            # Pass inputs to model (graph processing is now integrated)
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
        
        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)
            
        # Early stopping check with tolerance
        current_loss = sum_loss / len(pbar)
        
        # In distributed training, only rank 0 makes early stopping decisions
        # and broadcasts to other ranks
        early_stop = torch.tensor([0], device='cuda' if torch.cuda.is_available() else 'cpu')
        
        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            # Consider the model improved if loss decreased or within tolerance
            if current_loss < best_loss * (1 + tolerance):
                if current_loss < best_loss:
                    best_loss = current_loss
                patience_counter = 0
                logger.info(f"Loss improved or within tolerance: {current_loss:.6f} (best: {best_loss:.6f})")
            else:
                patience_counter += 1
                logger.info(f"Loss did not improve: {current_loss:.6f} (best: {best_loss:.6f}, patience: {patience_counter}/{config.patience})")
            
            # Set early stopping flag if patience exceeded
            if patience_counter >= config.patience:
                early_stop[0] = 1
        
        # Broadcast early stopping decision to all processes
        if engine.distributed:
            torch.distributed.broadcast(early_stop, 0)
            
        # Check if we should stop training
        if early_stop[0] == 1:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            
            # Save final checkpoint - only master process saves
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                              config.log_dir,
                                              config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                              config.log_dir,
                                              config.log_dir_link)
            
            # Delete stop flag file if it exists - only master process
            if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
                stop_flag = pathlib.Path("stop_training.flag")
                if stop_flag.exists():
                    try:
                        stop_flag.unlink()
                        logger.info("Deleted stop flag file.")
                    except Exception as e:
                        logger.warning(f"Could not delete stop flag file: {e}")
            
            # Sync all processes before exit in distributed mode
            if engine.distributed:
                logger.info("Synchronizing all processes before exit...")
                torch.distributed.barrier()
            
            logger.info("Early stopping triggered - exiting training.")
            # Use os._exit for a more forceful exit
            os._exit(0)

        # Save checkpoint at regular intervals or at the end
        if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)
        
        # Check for early stopping flag file
        if check_stop_flag(engine):
            logger.info("Stop flag detected! Saving checkpoint and stopping training...")
            
            # Save checkpoint
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                              config.log_dir,
                                              config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                              config.log_dir,
                                              config.log_dir_link)
            
            # Delete flag file after processing it - only master process
            if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
                stop_flag = pathlib.Path("stop_training.flag")
                if stop_flag.exists():
                    try:
                        stop_flag.unlink()
                        logger.info("Deleted stop flag file.")
                    except Exception as e:
                        logger.warning(f"Could not delete stop flag file: {e}")
            
            # Sync all processes before exit in distributed mode
            if engine.distributed:
                logger.info("Synchronizing all processes before exit...")
                torch.distributed.barrier()
                
            logger.info("Manual stop flag detected - exiting training.")
            # Use os._exit for a more forceful exit
            os._exit(0)
