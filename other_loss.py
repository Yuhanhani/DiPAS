import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)    # can be real numbers
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()

        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class DiceLoss_integer(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss_integer, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)    # can be real numbers
        inputs = torch.where(inputs >= 0.5, 1, 0)  # make the pred output also binary

        targets = targets.view(-1)

        intersection = (inputs * targets).sum()

        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

# inputs = torch.zeros([2, 2, 2])
# targets = torch.tensor([[[1,0], [0,1]],[[1,0], [1,0]]])
# print(inputs)
# print(targets)

# dl = DiceLoss()
# a = dl.forward(inputs, targets)
# print(a)

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)  # can be real numbers
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection # union = (inputs | targets).sum for integer number

        iou = (intersection + smooth) / (union + smooth)

        return 1 - iou

class IoULoss_integer(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss_integer, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)  # can be real numbers
        inputs = torch.where(inputs >= 0.5, 1, 0)  # make the pred output also binary

        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        # print(intersection)
        total = (inputs + targets).sum()
        union = total - intersection # union = (inputs | targets).sum for integer number
        # print(union)

        iou = (intersection + smooth) / (union + smooth)

        return 1 - iou