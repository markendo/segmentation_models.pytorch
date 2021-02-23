import torch.nn as nn

from . import base
from . import functional as F
from  .base import Activation

import numpy as np
import torch


class JaccardLoss(base.Loss):

    def __init__(self, eps=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(
            y_pr, y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )

class DiceLoss(base.Loss):

    def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )

class ClassAverageDiceLoss(base.Loss):

    def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt, smooth=1):
        y_pr = self.activation(y_pr)
        intersection = (y_pr * y_gt).sum(dim=(0, 2, 3))
        dice = (2.*intersection + smooth) / (y_pr.sum(dim=(0, 2, 3)) + y_gt.sum(dim=(0, 2, 3)) + smooth)
        dice_average = dice.mean()
        return 1 - dice_average


class BinaryCrossEntropyLoss(base.Loss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pr, y_gt):
        y_pr.detach()
        return nn.BCEWithLogitsLoss()(y_pr, y_gt)
        

class DistillationLoss(base.Loss):

    def __init__(self, eps=1., temp=6., **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.temp = temp
    
    def forward(self, y_pr, y_gt):
        return nn.BCELoss()(nn.Sigmoid()(y_pr / self.temp), nn.Sigmoid()(y_gt / self.temp))


class L1Loss(nn.L1Loss, base.Loss):
    pass


class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    pass


class BCELoss(nn.BCELoss, base.Loss):
    pass


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass
