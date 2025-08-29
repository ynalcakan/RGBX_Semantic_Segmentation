import os.path as osp
import os
import sys
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

import warnings
# Filter out the specific DDP warning about stride mismatches
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")

from config import config
from dataloader.dataloader import get_train_loader, ValPre
from models.builder import EncoderDecoder as segmodel
from dataloader.RGBXDataset import RGBXDataset
from utils.init_func import init_weight, group_weight # For weight decay of the model - optimizer
from utils.lr_policy import WarmUpPolyLR, StepLR, OneCycleLR, ReduceLROnPlateauLR, CyclicLR, CosineAnnealingWarmupLR, MultiStageLR, LinearIncreaseLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor # For distributed training
from utils.loss_opr import FocalLoss2d, RCELoss, BalanceLoss, berHuLoss, SigmoidFocalLoss, TopologyAwareLoss, ClassBalancedCELoss, BatchBalancedCELoss, MABalancedCELoss, MedianFreqCELoss, CannyEdgeLoss, SoftEdgeLoss
from utils.metric import hist_info, compute_score
from utils.transforms import normalize

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
logger = get_logger()

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
    elif criterion == 'CE_CannyEdgeLoss':
        criterion1 = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
        criterion2 = CannyEdgeLoss(ignore_index=config.background, reduction='mean')
        criterion = (criterion1, criterion2)
    elif criterion == 'CE_SoftEdgeLoss':
        criterion1 = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
        criterion2 = SoftEdgeLoss(ignore_index=config.background, reduction='mean')
        criterion = (criterion1, criterion2)
    elif criterion == 'Mask2FormerLoss':
        criterion = Mask2FormerLoss(num_classes=config.num_classes, ignore_index=config.background)
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
            
            # Make all parameters contiguous to avoid DDP stride mismatch warnings
            for param in model.parameters():
                if param.requires_grad and not param.is_contiguous():
                    param.data = param.data.contiguous()
            
            # Use bucket-view and allow unused params so DDP can handle dynamic graphs without stride mismatches
            model = DistributedDataParallel(
                model,
                device_ids=[engine.local_rank],
                output_device=engine.local_rank,
                find_unused_parameters=True
            )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model,optimizer=optimizer)

    # Prepare test/val dataset for per-epoch evaluation (no gradient, no training impact)
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
    test_dataset = RGBXDataset(data_setting, 'val', val_pre)

    def evaluate_miou(current_model):
        """Evaluate mIoU on the test/val split. Runs entirely without gradients."""
        was_training = current_model.training
        current_model.eval()
        device = torch.device("cuda", engine.local_rank) if torch.cuda.is_available() else torch.device("cpu")

        hist = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0

        with torch.no_grad():
            for idx in range(test_dataset.get_length()):
                sample = test_dataset[idx]
                img = sample['data']
                label = sample['label']
                modal_x = sample['modal_x']

                # Normalize inputs like in evaluation utils
                img = normalize(img, config.norm_mean, config.norm_std)
                if len(modal_x.shape) == 2:
                    modal_x = normalize(modal_x, 0, 1)
                    modal_x = np.expand_dims(modal_x, -1)
                    modal_x = np.repeat(modal_x, 3, axis=2)
                else:
                    modal_x = normalize(modal_x, config.norm_mean, config.norm_std)

                img_t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
                modal_x_t = torch.from_numpy(modal_x.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

                out = current_model(img_t, modal_x_t)

                # Support both standard decoders and Mask2Former outputs
                if isinstance(out, dict) and ('pred_logits' in out) and ('pred_masks' in out):
                    pred_logits = out['pred_logits']               # [B, Q, C+1]
                    pred_masks = out['pred_masks']                 # [B, Q, H, W]
                    class_probs = torch.softmax(pred_logits, dim=-1)[..., :config.num_classes]
                    mask_probs = torch.sigmoid(pred_masks)
                    score_map = torch.einsum('bqc,bqhw->bchw', class_probs, mask_probs)
                    pred = score_map.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.int32)
                else:
                    if isinstance(out, tuple):
                        out = out[0]
                    pred = out.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.int32)

                hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
                hist += hist_tmp
                labeled += labeled_tmp
                correct += correct_tmp

        iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
        if was_training:
            current_model.train()
        return float(mean_IoU)

    if engine.continue_state_object:
        engine.restore_checkpoint()
    else:
        optimizer.zero_grad()
        model.train()
        logger.info('begin trainning:')
    
    best_miou = -1.0
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

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            modal_xs = modal_xs.cuda(non_blocking=True)

            aux_rate = 0.2

            # Handle DDP-wrapped model when checking criterion
            _model = model.module if isinstance(model, DistributedDataParallel) else model
            # Decide based on the model's own criterion to avoid mismatch with decoder-specific loss
            if isinstance(_model.criterion, tuple):
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
                if isinstance(_model.criterion, tuple):
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
                if isinstance(_model.criterion, tuple):
                    # Initialize loss_sums dict if it doesn't exist
                    if not hasattr(engine, 'loss_sums'):
                        engine.loss_sums = {key: 0.0 for key in loss_components.keys()}
                    
                    for key in loss_components.keys():
                        engine.loss_sums[key] += loss_components[key].item()
                        print_str += ' %s=%.4f' % (key, engine.loss_sums[key]/(idx+1))

            del loss
            pbar.set_description(print_str, refresh=True)
        
        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)
            
            # Add individual loss components to TensorBoard if using multiple losses
            if isinstance(_model.criterion, tuple) and hasattr(engine, 'loss_sums'):
                for key in engine.loss_sums.keys():
                    tb.add_scalar(f'train_{key}', engine.loss_sums[key] / len(pbar), epoch)
                # Reset loss sums for next epoch
                engine.loss_sums = {key: 0.0 for key in engine.loss_sums.keys()}

        # Evaluate on test set (no gradients) and save best-by-mIoU checkpoint
        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            miou = evaluate_miou(model)
            if tb is not None:
                tb.add_scalar('test_mIoU', miou, epoch)
            logger.info(f"[Epoch {epoch}] Test mIoU: {miou:.4f}")
            if miou > best_miou:
                best_miou = miou
                os.makedirs(config.checkpoint_dir, exist_ok=True)
                best_ckpt = osp.join(config.checkpoint_dir, 'best_miou.pth')
                engine.save_checkpoint(best_ckpt)
                logger.info(f"Saved new best mIoU checkpoint at {best_ckpt} (mIoU={best_miou:.4f})")

        if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)
