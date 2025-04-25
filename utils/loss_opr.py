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

class TopologyAwareLoss(nn.Module):
    def __init__(self, ignore_index=255, reduction='mean', boundary_weight=1.0, connectivity_weight=0.1):
        super(TopologyAwareLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.boundary_weight = boundary_weight
        self.connectivity_weight = connectivity_weight
        
        # Laplacian kernel for boundary detection
        self.laplacian_kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).cuda()
        
    def get_boundary_map(self, tensor):
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(1)
        tensor = tensor.float()
        boundary = F.conv2d(tensor, self.laplacian_kernel, padding=1)
        boundary = torch.abs(boundary)
        boundary = (boundary > 0.1).float()
        return boundary
        
    def forward(self, pred, target):
        # Get predicted class probabilities
        pred_soft = F.softmax(pred, dim=1)
        num_classes = pred.size(1)
        
        # Create mask for valid pixels (not ignored)
        valid_mask = (target != self.ignore_index)
        
        # Mask out ignored pixels in target
        masked_target = target.clone()
        masked_target[~valid_mask] = 0
        
        # Create one-hot encoded target (only for valid pixels)
        target_one_hot = torch.zeros_like(pred_soft)
        for c in range(num_classes):
            class_mask = (masked_target == c) & valid_mask
            target_one_hot[:, c][class_mask] = 1
        
        # Calculate boundary loss
        boundary_loss = 0
        for i in range(num_classes):
            pred_boundary = self.get_boundary_map(pred_soft[:, i:i+1])
            target_boundary = self.get_boundary_map(target_one_hot[:, i:i+1])
            # Only compute loss for valid regions
            valid_boundary = valid_mask.unsqueeze(1).float()
            boundary_loss += F.binary_cross_entropy_with_logits(
                pred_boundary * valid_boundary,
                target_boundary * valid_boundary,
                reduction='sum'
            )
        boundary_loss = boundary_loss / (valid_mask.float().sum() + 1e-8)
        
        # Calculate connectivity loss
        connectivity_loss = 0
        for i in range(pred.size(0)):  # For each sample in batch
            for c in range(num_classes):
                # Skip ignored class
                if c == self.ignore_index:
                    continue
                    
                # Get masks for current class
                pred_mask = (pred_soft[i, c] > 0.5).float()
                target_mask = target_one_hot[i, c].float()
                
                # Skip if no target pixels for this class
                if target_mask.sum() == 0:
                    continue
                
                # Apply valid mask
                pred_mask = pred_mask * valid_mask[i].float()
                target_mask = target_mask * valid_mask[i].float()
                
                # Get connected components
                pred_components = self.get_connected_components(pred_mask)
                target_components = self.get_connected_components(target_mask)
                
                # Penalize difference in number of components
                connectivity_loss += torch.abs(pred_components - target_components)
        
        connectivity_loss = connectivity_loss / (pred.size(0) * num_classes + 1e-8)
        
        return self.boundary_weight * boundary_loss + self.connectivity_weight * connectivity_loss
    
    def get_connected_components(self, mask):
        # Convert to numpy for connected components analysis
        mask_np = mask.detach().cpu().numpy()
        _, num_components = nd.label(mask_np)
        return torch.tensor(num_components, device=mask.device).float()
    
class MedianFreqCELoss(nn.Module):
    """
    Median Frequency Balancing Cross Entropy Loss.
    Weights each class by the ratio of median frequency to class frequency.
    This is more robust to outliers than simple inverse frequency.
    """
    def _init_(self, num_classes=9, ignore_index=255):
        super(MedianFreqCELoss, self)._init_()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
    def forward(self, pred, target):
        # Compute frequencies in batch
        valid_mask = (target != self.ignore_index)
        freqs = torch.zeros(self.num_classes, device=pred.device)
        
        total_valid = valid_mask.sum().float()
        for c in range(self.num_classes):
            class_count = torch.sum((target == c) & valid_mask).float()
            freqs[c] = class_count / (total_valid + 1e-8)
        
        # Get median frequency
        median_freq = torch.median(freqs[freqs > 0])
        
        # Compute weights as median / frequency 
        weights = torch.zeros_like(freqs)
        nonzero_mask = freqs > 0
        weights[nonzero_mask] = median_freq / (freqs[nonzero_mask] + 1e-8)
        
        # Apply cross-entropy with weights
        loss = F.cross_entropy(pred, target, weight=weights, ignore_index=self.ignore_index)
        return loss
    
class DiceLoss(nn.Module):
    def __init__(self, num_classes=9, ignore_index=255, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, outputs, targets):
        """
        Parameters:
            outputs: predicted segmentation map [B, C, H, W], where C is the number of classes
            targets: ground truth segmentation map [B, H, W]
        """
        # Get predictions and targets
        pred_masks = outputs  # [B, C, H, W] - predicted class probabilities (logits)
        target_masks = targets  # [B, H, W] - ground truth labels
        # Create mask for valid pixels (not ignored)
        valid_mask = (target_masks != self.ignore_index)
        # Replace ignore_index in targets to class 0 to avoid out-of-bounds in one-hot
        masked_targets = target_masks.clone()
        masked_targets[~valid_mask] = 0  # map ignored labels to class 0
        target_masks = masked_targets
        # One-hot encode the target masks (for multi-class segmentation)
        target_masks = F.one_hot(target_masks, num_classes=self.num_classes).permute(0, 3, 1, 2).float()  # [B, C, H, W]
        # Zero out ignore_index pixels so they do not contribute
        valid_mask = valid_mask.unsqueeze(1).float()  # [B,1,H,W]
        target_masks = target_masks * valid_mask
        # Sigmoid the predictions
        pred_masks = torch.sigmoid(pred_masks)  # Apply sigmoid to get probabilities
        # Zero out ignored pixels in predictions
        pred_masks = pred_masks * valid_mask

        # Calculate Dice loss for each class
        dice_loss = 0
        for i in range(self.num_classes):
            # Select the i-th classw
            target_class = target_masks[:, i, :, :]
            pred_class = pred_masks[:, i, :, :]

            # Calculate intersection and union for Dice score
            intersection = (pred_class * target_class).sum(dim=(1, 2))
            union = pred_class.sum(dim=(1, 2)) + target_class.sum(dim=(1, 2))

            # Compute Dice score for this class
            dice_score = (2.0 * intersection + self.eps) / (union + self.eps)

            # Add Dice loss (1 - Dice score)
            dice_loss += (1 - dice_score).mean()

        return dice_loss / self.num_classes
    
class FocalDiceLoss(nn.Module):
    """
    Combined Focal + Dice loss with per-term weights.
      loss = alpha * focal_loss + beta * dice_loss
    """
    def __init__(self, alpha: float = 0.5, beta: float = 0.5,
                 num_classes: int = 9, ignore_index: int = 255):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.focal = FocalLoss2d(ignore_index=ignore_index, reduction='mean')
        self.dice  = DiceLoss(num_classes=num_classes,
                              ignore_index=ignore_index,
                              eps=1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            pred:   [B, C, H, W] raw logits
            target: [B, H, W] integer labels
        """
        f = self.focal(pred, target)
        d = self.dice(pred, target)
        return self.alpha * f + self.beta * d