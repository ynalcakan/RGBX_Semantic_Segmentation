import numpy as np
import scipy.ndimage as nd

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.logger import get_logger

logger = get_logger()

class FocalLoss2d(nn.Module):
    def __init__(self, gamma=0, weight=None, reduction='mean', ignore_index=255):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        if weight:
            self.loss = nn.NLLLoss(weight=torch.from_numpy(np.array(weight)).float(),
                                 reduction=reduction, ignore_index=ignore_index)
        else:
            self.loss = nn.NLLLoss(reduction=reduction, ignore_index=ignore_index)

    def forward(self, input, target):
        return self.loss((1 - F.softmax(input, 1))**2 * F.log_softmax(input, 1), target)


class RCELoss(nn.Module):
    def __init__(self, ignore_index=255, reduction='mean', weight=None, class_num=37, beta=0.01):
        super(RCELoss, self).__init__()
        self.beta = beta
        self.class_num = class_num
        self.ignore_label = ignore_index
        self.reduction = reduction
        self.criterion = nn.NLLLoss(reduction=reduction, ignore_index=ignore_index, weight=weight)
        self.criterion2 = nn.NLLLoss(reduction='none', ignore_index=ignore_index, weight=weight)

    def forward(self, pred, target):
        b, c, h, w = pred.shape
        max_pred, max_id = torch.max(pred, dim=1)		# pred (b, h, w)
        target_flat = target.view(b, 1, h, w)
        mask = (target_flat.ne(self.ignore_label)).float()
        target_flat = (mask * target_flat.float()).long()
        # convert to onehot
        label_pred = torch.zeros(b, self.class_num, h, w).cuda().scatter_(1, target_flat, 1)
        # print(label_pred.shape, max_id.shape)

        prob = torch.exp(pred)
        prob = F.softmax(prob, dim=1)      # i add this

        weighted_pred = F.log_softmax(pred, dim=1)
        loss1 = self.criterion(weighted_pred, target)

        label_pred = torch.clamp(label_pred, min=1e-9, max=1.0-1e-9)

        label_pred = torch.log(label_pred)
        loss2 = self.criterion2(label_pred, max_id)
        loss2 = torch.mean(loss2*mask)
        # print(loss1, loss2)
        loss = loss1 + self.beta*loss2
        # print(loss1, loss2)
        # print(loss)
        return loss

class BalanceLoss(nn.Module):
    def __init__(self, ignore_index=255, reduction='mean', weight=None):
        super(BalanceLoss, self).__init__()
        self.ignore_label = ignore_index
        self.reduction = reduction
        self.criterion = nn.NLLLoss(reduction=reduction, ignore_index=ignore_index, weight=weight)

    def forward(self, pred, target):
        # prob = torch.exp(pred)
        # # prob = F.softmax(prob, dim=1)      # i add this
        # weighted_pred = pred * (1 - prob) ** 2
        # loss = self.criterion(weighted_pred, target)

        prob = torch.exp(pred)
        prob = F.softmax(prob, dim=1)      # i add this
        weighted_pred = F.log_softmax(pred, dim=1) * (1 - prob) ** 2
        loss = self.criterion(weighted_pred, target)
        return loss

class berHuLoss(nn.Module):
    def __init__(self, delta=0.2, ignore_index=0, reduction='mean'):
        super(berHuLoss,self).__init__()
        self.delta = delta
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, pred, target):
        valid_mask = (1 - target.eq(self.ignore_index)).float()
        valid_delta = torch.abs(pred - target) * valid_mask
        max_delta = torch.max(valid_delta)
        delta = self.delta * max_delta

        f_mask = (1 - torch.gt(target, delta)).float() * valid_mask
        s_mask = (1 - f_mask ) * valid_mask
        f_delta =  valid_delta * f_mask
        s_delta = ((valid_delta **2) + delta **2) / (2 * delta) * s_mask

        loss = torch.mean(f_delta + s_delta)
        return loss


class SigmoidFocalLoss(nn.Module):
    def __init__(self, ignore_label, gamma=2.0, alpha=0.25, reduction='mean'):
        super(SigmoidFocalLoss, self).__init__()
        self.ignore_label = ignore_label
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        pred = pred.view(b, c, -1)  # B,C,H*W
        target = target.view(b, -1)  # B,H*W
        
        # Create a mask for valid pixels (not ignore_label)
        valid_mask = (target != self.ignore_label).float()
        
        # Clip target values to be within the valid range
        target = torch.clamp(target, 0, c - 1)
        
        # Convert target to one-hot encoding
        target_one_hot = F.one_hot(target, num_classes=c).float().permute(0, 2, 1)
        
        # Calculate probabilities
        probs = torch.sigmoid(pred)
        pt = torch.where(target_one_hot == 1, probs, 1 - probs)
        
        # Calculate focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Calculate alpha weight
        alpha_weight = torch.where(target_one_hot == 1, self.alpha * torch.ones_like(probs), (1 - self.alpha) * torch.ones_like(probs))
        
        # Calculate loss
        loss = -alpha_weight * focal_weight * torch.log(pt + 1e-8)
        
        # Apply valid mask
        loss = loss * valid_mask.unsqueeze(1)
        
        # Reduce loss
        if self.reduction == 'mean':
            return loss.sum() / (valid_mask.sum() + 1e-8)
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss.sum(1)


class ProbOhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label, reduction='mean', thresh=0.6, min_kept=256,
                 down_ratio=1, use_weight=False):
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       weight=weight,
                                                       ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            logger.info('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(1 - valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)     # 概率小于阈值的挖出来
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask
                # logger.info('Valid Mask: {}'.format(valid_mask.sum()))

        target = target.masked_fill_(1 - valid_mask, self.ignore_label)
        target = target.view(b, h, w)

        return self.criterion(pred, target)

class Mask2FormerLoss(nn.Module):
    def __init__(self, num_classes, matcher_weight_dict={'class': 2, 'mask': 5, 'dice': 5}, 
                 losses=['labels', 'masks'], eos_coef=0.1, ignore_index=255):
        """
        Parameters:
            num_classes: number of object classes (including background)
            matcher_weight_dict: weights for matcher cost computation
            losses: list of losses to apply
            eos_coef: relative weight for no-object class
            ignore_index: label value to ignore
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher_weight_dict = matcher_weight_dict
        self.losses = losses
        self.eos_coef = eos_coef
        self.ignore_index = ignore_index
        
        # Create class weight for cross entropy
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef  # lower weight for no-object class
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices):
        """Classification loss (NLL)"""
        src_logits = outputs['pred_logits']  # [B, num_queries, num_classes+1]
        
        # Flatten the targets to match with predictions
        B, num_queries, _ = src_logits.shape
        target_classes = torch.full((B, num_queries), self.num_classes,
                                  dtype=torch.int64, device=src_logits.device)
        
        # Create a binary mask for valid (non-ignored) pixels
        valid_mask = targets != self.ignore_index  # [B, H, W]
        
        # For each valid pixel in the target, assign it to the closest query
        for b in range(B):
            valid_pixels = valid_mask[b]  # [H, W]
            if valid_pixels.any():
                # Get valid target values
                valid_targets = targets[b][valid_pixels]  # [N]
                
                # Get predictions for this batch
                pred_masks = outputs['pred_masks'][b]  # [num_queries, H, W]
                pred_masks = pred_masks[:, valid_pixels]  # [num_queries, N]
                
                # Compute similarity between queries and target pixels
                similarity = pred_masks.sigmoid()  # [num_queries, N]
                
                # Assign each pixel to the most similar query
                assignments = similarity.max(dim=0)[1]  # [N]
                
                # Update target classes for assigned queries
                for query_idx in range(num_queries):
                    query_pixels = assignments == query_idx
                    if query_pixels.any():
                        # Most common class for this query's assigned pixels
                        target_class = valid_targets[query_pixels].mode()[0]
                        target_classes[b, query_idx] = target_class
        
        # Add focal loss
        ce_loss = F.cross_entropy(src_logits.transpose(1, 2), target_classes, 
                                self.empty_weight, ignore_index=self.num_classes,
                                reduction='none')
        p = torch.exp(-ce_loss)
        loss_ce = ((1 - p) ** 2.0) * ce_loss
        loss_ce = loss_ce.mean()
        
        return loss_ce

    def loss_masks(self, outputs, targets, indices):
        """Compute the losses related to the masks"""
        src_masks = outputs['pred_masks']  # [B, Q, H, W]
        B, Q, H, W = src_masks.shape

        # Cross entropy loss - reshape masks to [B*H*W, Q]
        src_masks_ce = src_masks.permute(0, 2, 3, 1).reshape(-1, Q)  # [B*H*W, Q]
        targets_ce = targets.reshape(-1)  # [B*H*W]
        
        # Don't use class weights for mask loss
        ce_loss = F.cross_entropy(src_masks_ce, targets_ce, 
                                weight=None,  # Remove empty_weight here
                                ignore_index=self.ignore_index,
                                reduction='mean')

        # Dice loss
        target_masks = (targets.unsqueeze(1) == torch.arange(self.num_classes, 
                       device=targets.device).reshape(1, -1, 1, 1))
        target_masks = target_masks.float()
        
        valid_mask = (targets != self.ignore_index).unsqueeze(1).float()
        
        src_masks = src_masks.sigmoid()
        dice_loss = 0
        
        for i in range(self.num_classes):
            if target_masks[:, i].sum() > 0:
                dice_score = 2 * (src_masks * target_masks[:, i].unsqueeze(1) * valid_mask).sum(dim=(2, 3)) / \
                           (src_masks.sum(dim=(2, 3)) + target_masks[:, i].unsqueeze(1).sum(dim=(2, 3)) + 1e-8)
                dice_loss += (1 - dice_score.mean())
        
        dice_loss = dice_loss / self.num_classes

        # Combine losses with their respective weights
        combined_loss = self.matcher_weight_dict['mask'] * ce_loss + \
                       self.matcher_weight_dict['dice'] * dice_loss
        
        return combined_loss

    def forward(self, outputs, targets):
        """
        Parameters:
            outputs: dict containing 'pred_logits' and 'pred_masks'
            targets: ground truth segmentation map [B, H, W]
        """
        losses = {}
        
        if 'labels' in self.losses:
            losses['loss_cls'] = self.loss_labels(outputs, targets, None) * self.matcher_weight_dict['class']
        
        if 'masks' in self.losses:
            losses['loss_mask'] = self.loss_masks(outputs, targets, None)
        
        # Compute total loss as weighted sum
        total_loss = sum(losses.values())
        
        return total_loss
