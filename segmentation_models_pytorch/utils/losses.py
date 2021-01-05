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

class MyDiceLoss(base.Loss):

    def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt, smooth=1):
        #Class Averaged Dice
        # y_pr = self.activation(y_pr)
        # intersection = (y_pr * y_gt).sum(dim=(0, 2, 3))
        # dice = (2.*intersection + smooth) / (y_pr.sum(dim=(0, 2, 3)) + y_gt.sum(dim=(0, 2, 3)) + smooth)
        # dice_average = dice.mean()
        # return 1 - dice_average

        #Dice BCE
        # inputs = self.activation(y_pr)     
        # inputs = inputs.view(-1)
        # targets = y_gt.view(-1)
        # intersection = (inputs * targets).sum()                            
        # dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        # BCE = torch.nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')
        # Dice_BCE = BCE + dice_loss
        # return Dice_BCE

        #Dice
        y_pr = self.activation(y_pr)
        y_pr = y_pr.view(-1)
        y_gt = y_gt.view(-1)
        intersection = (y_pr * y_gt).sum()
        dice = (2.*intersection + smooth)/(y_pr.sum() + y_gt.sum() + smooth)  
        return 1 - dice

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
        #from peterlht (classification)
        # KD_loss = nn.KLDivLoss(reduction='batchmean')(nn.functional.log_softmax(y_pr/self.temp, dim=1), nn.functional.softmax(y_gt/self.temp, dim=1))
        # return KD_loss

        #from irfan structure knowledge distillation criterion pixel wise
        # y_pr.detach()
        # assert y_pr.shape == y_gt.shape, 'the output dim of teacher and student differ'
        # N,C,W,H = y_pr.shape
        # softmax_pred_T = nn.functional.softmax(y_gt.permute(0, 2, 3, 1).contiguous().view(-1, C), dim=1)
        # logsoftmax = nn.LogSoftmax(dim=1)
        # loss = (torch.sum( - softmax_pred_T * logsoftmax(y_pr.permute(0,2,3,1).contiguous().view(-1, C))))/W/H
        # return loss

        # CATEGORICAL CROSS ENTROPY WITH TEMP (altered from irfan criterion pixel wise)
        # y_pr.detach()
        # assert y_pr.shape == y_gt.shape, 'the output dim of teacher and student differ'
        # N,C,W,H = y_pr.shape
        # softmax_pred_T = nn.functional.softmax(y_gt.permute(0,2,3,1).contiguous().view(-1, C) / self.temp, dim=1)
        # log_softmax_pred_S = nn.functional.log_softmax(y_pr.permute(0,2,3,1).contiguous().view(-1, C) / self.temp, dim=1)
        # loss = -torch.sum(softmax_pred_T * log_softmax_pred_S)/W/H
        # return loss

        # BINARY CROSS ENTROPY
        # y_pr.detach()
        # assert y_pr.shape == y_gt.shape, 'the output dim of teacher and student differ'
        # N,C,W,H = y_pr.shape
        # sigmoid_pred_T = nn.Sigmoid()(y_gt.flatten())
        # sigmoid_n_pred_T = 1 - sigmoid_pred_T
        # sigmoid_pred_S = nn.Sigmoid()(y_pr.flatten())
        # sigmoid_n_pred_S = 1 - sigmoid_pred_S
        # log_sigmoid_pred_S = torch.log(sigmoid_pred_S)
        # log_sigmoid_n_pred_S = torch.log(sigmoid_n_pred_S)
        # loss = -torch.sum((sigmoid_pred_T * log_sigmoid_pred_S)+(sigmoid_n_pred_T * log_sigmoid_n_pred_S))/W/H
        # return loss

        # PYTORCH BCE
        # y_pr.detach()
        return nn.BCELoss()(nn.Sigmoid()(y_pr / self.temp), nn.Sigmoid()(y_gt / self.temp))

        # Updated Categorical Cross Entropy with temp (something wrong, performs worse than original)
        # y_pr.detach()
        # softmax_pred_T = nn.Softmax(dim=1)(y_gt / self.temp)
        # log_softmax_pred_S = nn.LogSoftmax(dim=1)(y_pr / self.temp)
        # return -torch.mean(softmax_pred_T * log_softmax_pred_S)

        #dice loss
        # y_gt = nn.Sigmoid()(y_gt)
        # return 1 - F.f_score(
        #     y_pr, y_gt,
        #     beta=1.,
        #     eps=1.,
        #     threshold=None,
        #     ignore_channels=None,
        # )




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
