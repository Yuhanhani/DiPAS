import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchmetrics.classification import BinaryJaccardIndex
# from torchmetrics.classification import BinaryF1Score


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

        return dice


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

        return dice


# dl = performance_metric.DiceLoss()
# e = dl.forward(b, a)
# print(e)

# dl = performance_metric.DiceLoss_integer()
# e = dl.forward(b, a)
# print(e)

# f_1 = BinaryF1Score()  # this and second will be larger generally than first one, but all are different (second has smoothing)
# f = f_1(b, a)
# print(f)


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

        return iou

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

        return iou

# dl = performance_metric.IoULoss()
# e = dl.forward(b, a)
# print(e)

# b = torch.tensor([[0.1, 0.8],[0.9,0.2]])
# a = torch.tensor([[0, 0], [1, 0]])
# dl = IoULoss_integer()
# e1 = dl.forward(b, a)
# print(e1)
#
# b = torch.tensor([[0.7, 0.2],[0.1,0.6]])
# a = torch.tensor([[0, 1], [0, 0]])
# dl = IoULoss_integer()
# e2 = dl.forward(b, a)
# print(e2)
#
# print(0.5*e1+0.5*e2)
#
# b = torch.tensor([[[0.1, 0.8],[0.9,0.2]], [[0.7, 0.2],[0.1,0.6]]])
# a = torch.tensor([[[0,0],[1,0]],[[0, 1], [0, 0]]])
# dl = IoULoss_integer()
# e2 = dl.forward(b, a)
# print(e2)



# jaccard = BinaryJaccardIndex()
# f = jaccard(b, a)
# print(f)


