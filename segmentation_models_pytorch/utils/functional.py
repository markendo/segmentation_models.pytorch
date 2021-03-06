import torch
import numpy as np


def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x

def _taskwise_threshold(x, thresholds):
    thresholds_tensor = torch.FloatTensor(thresholds).to(device='cuda')
    x_sub = x - thresholds_tensor[None,:,None,None]
    return _threshold(x_sub, threshold=0)

def _mask_thresholds(x, thresholds):
    thresholds_tensor = torch.FloatTensor(thresholds).to(device='cuda')
    x_dup = x.repeat(len(thresholds),1,1,1,1)
    x_sub = x_dup - thresholds_tensor[:, None, None, None, None]
    return _threshold(x_sub, threshold=0)

def iou_taskwise_thresholds(pr, gt, iou, thresholds):
    """
    Calculate Intersection over Union between ground truth and prediction
    y_pr masked using thresholds list, which contains optimal threshold
    for each class

    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        iou: current iou score
        thresholds (list): class-specific threshold to mask y_[r]
    Returns:
        float: IoU (Jaccard) score
    """
    pr = _taskwise_threshold(pr, thresholds)
    intersection = torch.sum(gt * pr, dim=(0, 2, 3)).cpu().detach().numpy()
    union = torch.sum(gt, dim=(0, 2, 3)).cpu().detach().numpy() + torch.sum(pr, dim=(0, 2, 3)).cpu().detach().numpy() - intersection
    return (iou[0] + intersection, iou[1] + union)

def iou(pr, gt, iou, thresholds):
    """
    Calculate Intersection over Union between ground truth and prediction
    iou calculated for all possible y_pr mask thresholds in thresholds. Caller function
    can then pick best iou for each class

    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        iou: current iou score for different thresholds
        thresholds (list): different thresholds to mask y_pr
    Returns:
        float: IoU (Jaccard) score
    """
    # try out all thresholds in thresholds array
    pr = _mask_thresholds(pr, thresholds)
    gt = gt.repeat(len(thresholds),1,1,1,1)
    intersection = torch.sum(gt * pr, dim=(1, 3, 4)).cpu().detach().numpy()
    union = torch.sum(gt, dim=(1, 3, 4)).cpu().detach().numpy() + torch.sum(pr, dim=(1, 3, 4)).cpu().detach().numpy() - intersection
    return iou[0] + intersection, iou[1] + union

jaccard = iou


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score


def accuracy(pr, gt, threshold=0.5, ignore_channels=None):
    """Calculate accuracy score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt == pr, dtype=pr.dtype)
    score = tp / gt.view(-1).shape[0]
    return score


def precision(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate precision score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp

    score = (tp + eps) / (tp + fp + eps)

    return score


def recall(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Recall between ground truth and prediction
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: recall score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fn = torch.sum(gt) - tp

    score = (tp + eps) / (tp + fn + eps)

    return score
