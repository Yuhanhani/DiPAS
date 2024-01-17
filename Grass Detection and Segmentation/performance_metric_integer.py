
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F



class DiceLoss_integer(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss_integer, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = inputs.view(-1)
        inputs = torch.where(inputs >= 0.5, 1, 0)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()

        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return dice




class IoULoss_integer(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss_integer, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = inputs.view(-1)
        inputs = torch.where(inputs >= 0.5, 1, 0)

        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 

        iou = (intersection + smooth) / (union + smooth)

        return iou
