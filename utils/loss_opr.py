import numpy as np
import scipy.ndimage as nd

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from engine.logger import get_logger

logger = get_logger()

class FocalLoss2d(nn.Module):
    """
    FocalLoss2d is a loss function that combines the cross-entropy loss and the focal loss.
    """
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
    """
    RCELoss is a loss function that combines the cross-entropy loss and the RCE loss.
    """
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
    """
    BalanceLoss is a loss function that combines the cross-entropy loss and the balance loss.
    """
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
    """
    berHuLoss is a loss function that combines the cross-entropy loss and the berHu loss.
    """
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
    """
    SigmoidFocalLoss is a loss function that combines the cross-entropy loss and the sigmoid focal loss.
    """
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
    """
    ProbOhemCrossEntropy2d is a loss function that combines the cross-entropy loss and the probability-based OHEM loss.
    """
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
    """
    Semantic Mask2Former-style loss.

    Aggregates query masks into per-class probability maps and applies:
    - per-pixel NLLLoss (cross-entropy) against class labels
    - per-class Dice loss on aggregated maps

    This avoids instance-level matching and works well for semantic segmentation.
    """
    def __init__(self, num_classes, matcher_weight_dict={'class': 2.0, 'dice': 5.0}, ignore_index=255, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps
        # reuse keys from existing config for compatibility
        self.class_weight = matcher_weight_dict.get('class', 2.0)
        self.dice_weight = matcher_weight_dict.get('dice', 5.0)

    def _aggregate_class_maps(self, pred_logits: torch.Tensor, pred_masks: torch.Tensor) -> torch.Tensor:
        """
        pred_logits: [B, Q, C+1] (includes no-object)
        pred_masks:  [B, Q, H, W] (logits)
        returns per-class probability maps [B, C, H, W]
        """
        B, Q, _ = pred_logits.shape
        _, _, H, W = pred_masks.shape

        # Query class probabilities (exclude no-object)
        class_prob = F.softmax(pred_logits, dim=-1)[..., :self.num_classes]  # [B, Q, C]
        # Mask probabilities
        mask_prob = pred_masks.sigmoid()  # [B, Q, H, W]

        # Aggregate: S[b,c,h,w] = sum_q P(c|q) * P(mask_q at (h,w))
        # Implement via einsum for clarity
        # class_prob: [B,Q,C] -> [B,C,Q] for einsum convenience
        class_prob_t = class_prob.permute(0, 2, 1)  # [B, C, Q]
        mask_prob_flat = mask_prob.view(B, Q, H * W)  # [B, Q, HW]
        agg_flat = torch.einsum('bcq,bqh->bch', class_prob_t, mask_prob_flat)  # [B, C, HW]
        agg = agg_flat.view(B, self.num_classes, H, W)  # [B, C, H, W]

        # Normalize across queries magnitude by total mask mass to stabilize
        denom_flat = mask_prob_flat.sum(dim=1, keepdim=False)  # [B, HW]
        denom = denom_flat.view(B, 1, H, W) + self.eps
        agg = agg / denom  # still not normalized across classes

        # Normalize across classes to produce a valid per-pixel distribution
        class_sum = agg.sum(dim=1, keepdim=True) + self.eps
        prob = agg / class_sum  # [B, C, H, W]
        return prob

    def _dice_loss(self, pred_prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-class soft Dice loss on aggregated class probabilities.
        pred_prob: [B, C, H, W], target: [B, H, W] (int)
        """
        B, C, H, W = pred_prob.shape
        valid = (target != self.ignore_index).float().unsqueeze(1)  # [B,1,H,W]

        # one-hot target
        target_oh = torch.zeros((B, C, H, W), device=pred_prob.device, dtype=pred_prob.dtype)
        for c in range(C):
            target_oh[:, c] = (target == c).float()

        pred = pred_prob * valid
        targ = target_oh * valid

        pred_flat = pred.flatten(2)  # [B,C,HW]
        targ_flat = targ.flatten(2)

        intersection = (pred_flat * targ_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + targ_flat.sum(dim=2) + self.eps
        dice = 2.0 * intersection / union  # [B,C]
        # average over present classes to avoid penalizing absent classes too much
        present = (targ_flat.sum(dim=2) > 0).float()
        dice_per_batch = (dice * present).sum(dim=1) / (present.sum(dim=1) + self.eps)
        loss = 1.0 - dice_per_batch.mean()
        return loss

    def forward(self, outputs, targets):
        """
        outputs: { 'pred_logits': [B,Q,C+1], 'pred_masks': [B,Q,H,W] logits }
        targets: [B,H,W] int labels
        """
        pred_logits = outputs['pred_logits']
        pred_masks = outputs['pred_masks']

        # Aggregate queries into per-pixel per-class probabilities
        prob_maps = self._aggregate_class_maps(pred_logits, pred_masks)  # [B,C,H,W]

        # Cross-entropy on aggregated distribution
        log_prob = torch.log(prob_maps.clamp(min=self.eps))  # [B,C,H,W]
        ce = F.nll_loss(log_prob, targets, ignore_index=self.ignore_index, reduction='mean')

        # Dice loss per class
        dice = self._dice_loss(prob_maps, targets)

        return self.class_weight * ce + self.dice_weight * dice

class TopologyAwareLoss(nn.Module):
    """
    TopologyAwareLoss is a loss function that combines the cross-entropy loss and the topology-aware loss.
    """
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
    

class ClassBalancedCELoss(nn.Module):
    """
    ClassBalancedCELoss is a loss function that combines the cross-entropy loss and the class-balanced CE loss.
    """
    def __init__(self, samples_per_cls, beta=0.9999, ignore_index=255, reduction='mean'):
        super(ClassBalancedCELoss, self).__init__()
        self.beta = beta
        self.ignore_index = ignore_index
        self.reduction = reduction
        # compute effective number of samples per class
        effective_num = 1.0 - np.power(self.beta, samples_per_cls)
        weights = (1.0 - self.beta) / (effective_num + 1e-8)
        # normalize weights so that sum(weights) = num_classes
        weights = weights / np.sum(weights) * len(samples_per_cls)
        # register weight tensor
        weight_tensor = torch.from_numpy(weights).float()
        self.register_buffer('weight', weight_tensor)
        # define criterion using class-balanced weights
        self.criterion = nn.NLLLoss(weight=self.weight, reduction=self.reduction, ignore_index=self.ignore_index)

    def forward(self, pred, target):
        """Compute the class-balanced cross-entropy loss."""
        # pred: [N, C, H, W], target: [N, H, W]
        log_prob = F.log_softmax(pred, dim=1)
        loss = self.criterion(log_prob, target)
        return loss

class BatchBalancedCELoss(nn.Module):
    """
    BatchBalancedCELoss computes class weights adaptively from the current batch.
    This avoids the need to pre-compute class frequencies over the entire dataset.
    """
    def __init__(self, num_classes=9, ignore_index=255, reduction='mean'):
        super(BatchBalancedCELoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def forward(self, pred, target):
        # Compute class weights from current batch
        batch_size, _, h, w = pred.size()
        valid_mask = (target != self.ignore_index)
        
        # Count instances of each class in this batch
        weights = torch.zeros(self.num_classes, device=pred.device)
        for c in range(self.num_classes):
            class_pixels = torch.sum((target == c) & valid_mask).float()
            weights[c] = class_pixels + 1e-10  # avoid division by zero
        
        # Inverse frequency weighting
        weights = 1.0 / weights
        # Normalize weights
        weights = weights / weights.sum() * self.num_classes
        
        # Apply cross-entropy with computed weights
        loss = F.cross_entropy(pred, target, weight=weights, 
                              ignore_index=self.ignore_index, reduction=self.reduction)
        return loss

class MABalancedCELoss(nn.Module):
    """
    Moving Average Balanced Cross Entropy Loss.
    Maintains a moving average of class frequencies across batches for more stable weights.
    """
    def __init__(self, num_classes=9, ignore_index=255, momentum=0.9):
        super(MABalancedCELoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.momentum = momentum
        # Initialize with equal weights
        self.register_buffer('class_counts', torch.ones(num_classes))
        
    def forward(self, pred, target):
        # Update class counts with moving average
        valid_mask = (target != self.ignore_index)
        batch_counts = torch.zeros(self.num_classes, device=pred.device)
        
        for c in range(self.num_classes):
            batch_counts[c] = torch.sum((target == c) & valid_mask).float() + 1e-10
            
        # Update moving average
        self.class_counts = self.momentum * self.class_counts + (1 - self.momentum) * batch_counts
        
        # Compute weights as inverse frequency
        weights = 1.0 / self.class_counts
        weights = weights / weights.sum() * self.num_classes
        
        # Apply cross-entropy with updated weights
        loss = F.cross_entropy(pred, target, weight=weights, ignore_index=self.ignore_index)
        return loss

class MedianFreqCELoss(nn.Module):
    """
    Median Frequency Balancing Cross Entropy Loss.
    Weights each class by the ratio of median frequency to class frequency.
    This is more robust to outliers than simple inverse frequency.
    """
    def __init__(self, num_classes=9, ignore_index=255):
        super(MedianFreqCELoss, self).__init__()
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
    
class CannyEdgeLoss(nn.Module):
    def __init__(self, ignore_index=255, reduction='mean'):
        super(CannyEdgeLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, pred, target):
        # pred: [N, C, H, W] logits, target: [N, H, W]
        # Convert logits to discrete mask
        pred_mask = torch.argmax(pred, dim=1).cpu().numpy().astype(np.uint8)  # [N, H, W]
        target_np = target.cpu().numpy().astype(np.uint8)  # [N, H, W]

        batch_size = pred_mask.shape[0]
        loss_list = []
        for i in range(batch_size):
            # Compute edges using Canny on single-channel uint8 images
            pe = cv2.Canny(pred_mask[i], 10, 40) / 255.0    # now in {0,1}
            te = cv2.Canny(target_np[i],   10, 40) / 255.0

            pe_t = torch.from_numpy(pe).to(pred.device).unsqueeze(0).unsqueeze(0)
            te_t = torch.from_numpy(te).to(pred.device).unsqueeze(0).unsqueeze(0)

            loss_list.append(
                F.binary_cross_entropy(pe_t, te_t, reduction=self.reduction)
            )
        # Average over batch
        return torch.stack(loss_list).mean()
    
class SoftEdgeLoss(nn.Module):
    def __init__(self, ignore_index=255, reduction='mean'):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

        # Sobel kernels for horizontal & vertical gradients
        kernel_x = torch.tensor([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]], dtype=torch.float32)
        kernel_y = kernel_x.t()
        # We’ll apply them on a single‐channel mask, so shape is (1,1,3,3)
        self.register_buffer('kx', kernel_x.view(1,1,3,3))
        self.register_buffer('ky', kernel_y.view(1,1,3,3))

    def forward(self, pred, target):
        """
        pred: [N, C, H, W]  — raw logits
        target: [N, H, W]   — integer mask
        """
        #    Convert logits to per‐pixel class probabilities
        prob = F.softmax(pred, dim=1)                       # [N, C, H, W]
        #    Collapse to a single‐channel “label magnitude” map
        #    using either a soft-argmax or hard argmax—but keep it differentiable:
        #    Here we take the max prob per pixel (still gives a gradient via prob).
        max_prob, _ = prob.max(dim=1, keepdim=True)         # [N, 1, H, W]

        # Move kernels to the same device as input
        dev = max_prob.device
        kx = self.kx.to(dev)
        ky = self.ky.to(dev)
        gx = F.conv2d(max_prob, kx, padding=1)         # [N,1,H,W]
        gy = F.conv2d(max_prob, ky, padding=1)
        pred_edge = torch.sqrt(gx*gx + gy*gy + 1e-6)        # smooth L2 magnitude

        #   Same for the GT mask (treat as float image)
        t = target.unsqueeze(1).float()                     # [N,1,H,W]
        # ensure kernels on correct device
        t = t.to(dev)
        gx_t = F.conv2d(t, kx, padding=1)
        gy_t = F.conv2d(t, ky, padding=1)
        gt_edge = torch.sqrt(gx_t*gx_t + gy_t*gy_t + 1e-6)

        #    Finally compare edge maps with a simple L1 or BCE loss
        #    Here L1 works well for continuous maps:
        loss = F.l1_loss(pred_edge, gt_edge, reduction=self.reduction)
        return loss