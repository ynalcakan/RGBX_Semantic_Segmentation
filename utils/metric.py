# encoding: utf-8

import numpy as np

np.seterr(divide='ignore', invalid='ignore')


def hist_info(n_cl, pred, gt):
    assert (pred.shape == gt.shape)
    k = (gt >= 0) & (gt < n_cl)
    labeled = np.sum(k)
    correct = np.sum((pred[k] == gt[k]))
    confusionMatrix = np.bincount(n_cl * gt[k].astype(int) + pred[k].astype(int),
                        minlength=n_cl ** 2).reshape(n_cl, n_cl)
    return confusionMatrix, labeled, correct

def compute_score(hist, correct, labeled):
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IoU = np.nanmean(iou)
    mean_IoU_no_back = np.nanmean(iou[1:]) # useless for NYUDv2

    freq = hist.sum(1) / hist.sum()
    freq_IoU = (iou[freq > 0] * freq[freq > 0]).sum()

    classAcc = np.diag(hist) / hist.sum(axis=1)
    mean_pixel_acc = np.nanmean(classAcc)

    pixel_acc = correct / labeled

    return iou, mean_IoU, mean_IoU_no_back, freq_IoU, mean_pixel_acc, pixel_acc

def compute_weighted_score(hist, correct, labeled, class_weights):
    """
    Compute IoU and pixel accuracy metrics weighted by class_weights.
    Returns: iou, weighted_iou, mean_IoU, weighted_mean_IoU, freq_IoU, mean_pixel_acc, weighted_mean_pixel_acc, pixel_acc
    """
    # Compute standard metrics
    iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
    # Normalize class weights
    weights = np.array(class_weights, dtype=np.float32)
    norm_weights = weights / weights.sum()
    # Compute weighted IoU per class
    weighted_iou = iou * norm_weights
    # Weighted mean IoU
    weighted_mean_IoU = np.nansum(weighted_iou)
    # Compute per-class accuracy
    class_acc = np.diag(hist) / hist.sum(axis=1)
    # Weighted mean pixel accuracy
    weighted_mean_pixel_acc = np.nansum(class_acc * norm_weights)
    return iou, weighted_iou, mean_IoU, weighted_mean_IoU, freq_IoU, mean_pixel_acc, weighted_mean_pixel_acc, pixel_acc