
import os
from PIL import Image
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torchvision.models as models
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch
from torch import nn
import pandas as pd
import csv
import cv2
from sklearn.model_selection import train_test_split
import random
import sklearn
import data_augmentation
import dataset
import other_loss
import performance_metric

pd.options.display.max_rows = 1000

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# Download and read labels and data ------------------------------------------------------------------------------------

if torch.cuda.is_available():
    img_path = 'file path'
else:
    img_path = 'file path'

if torch.cuda.is_available():
    mask_path = 'file path'
else:
    mask_path = 'file path'

# Create dataloader ----------------------------------------------------------------------------------------------------

transform_1 = data_augmentation.data_augmentation_transform(phase='train')
train_dataset = dataset.CustomImageDataset(img_path=img_path, mask_path=mask_path, transform=transform_1)
transform_2 = data_augmentation.data_augmentation_transform(phase='test')
test_dataset = dataset.CustomImageDataset(img_path=img_path, mask_path=mask_path, transform=transform_2)

random.seed(1)
index = np.arange(100)
random.shuffle(index)
print(index)
split_index = np.split(index, 5)  # in order
print(split_index)

overall_dice_score = []
overall_dice_score_integer = []
overall_iou_score = []
overall_iou_score_integer = []

for k in range(5):

    test_index = split_index[k]

    test_data = []
    test_label = []
    train_data = []
    train_label = []

    for i in range(100):

        if i in test_index:
            image, mask = test_dataset.__getitem__(i)
            test_data.append([image, mask])
            print(f'{i} is in test')
        else:
            image, mask = train_dataset.__getitem__(i)
            train_data.append([image, mask])
            print(f'{i} is in train')

    print(len(train_data))
    print(len(test_data))


    train_batch_size = 7
    train_dataloader = DataLoader(train_data, batch_size=train_batch_size)
    test_batch_size = 1
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size)

    # Use model-------------------------------------------------------------------------------------------------------------

    # model = models.segmentation.deeplabv3_resnet101(weights='DeepLabV3_ResNet101_Weights.DEFAULT', weights_backbone='ResNet101_Weights.DEFAULT')
    # model.classifier = DeepLabHead(2048, 1)

    # model = models.segmentation.fcn_resnet50(weights='FCN_ResNet50_Weights.DEFAULT', weights_backbone='ResNet50_Weights.DEFAULT')
    # model.classifier = FCNHead(2048, 1)

    model = model.to(device)

    # Define all hyperparameters--------------------------------------------------------------------------------------------

    learning_rate = 0.01  # set the initial learning rate
    epochs = 15
    gamma = 0.9

    loss_BCE = nn.BCELoss(reduction='mean')
    loss_DICE = other_loss.DiceLoss_integer()
    loss_IoU = other_loss.IoULoss_integer()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1500)

    # Define the training loop ---------------------------------------------------------------------------------------------

    def train_loop(dataloader, model, loss_fn_1, loss_fn_2, optimizer):

        size = len(dataloader.dataset)
        current = 0
        total_loss = 0

        for batch, (X, y) in enumerate(dataloader):


            X = X.to(device)  # float tensor, 1, 3, 572, 572
            y = y.to(device)  # long tensor, 1, 1, 572, 572

            # forward propagation
            pred = torch.sigmoid(model(X))  # float tensor, 1, 1, 572, 572

            loss = loss_fn_1(pred, y) + loss_fn_2.forward(y, pred)  # target should be of type long tensor


            # Backward propagation

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            loss, current = loss.item(), current + len(X)

            print(f'loss:{loss:>7f} [{current:>5d}/{size:>5d}]')

            total_loss = total_loss + loss

            local_dice_score = performance_metric.DiceLoss()
            local_dice_score = local_dice_score.forward(pred, y)
            # print(f'local_dice_score:{local_dice_score}')

            local_dice_score_integer = performance_metric.DiceLoss_integer()
            local_dice_score_integer = local_dice_score_integer.forward(pred, y)
            # print(f'local_dice_score_integer:{local_dice_score_integer}')

            local_iou_score = performance_metric.IoULoss()
            local_iou_score = local_iou_score.forward(pred, y)
            # print(f'local_iou_score:{local_iou_score}')

            local_iou_score_integer = performance_metric.IoULoss_integer()
            local_iou_score_integer = local_iou_score_integer.forward(pred, y)
            # print(f'local_iou_score_integer:{local_iou_score_integer}')

            if batch == 0:
                dice_score = local_dice_score
                dice_score_integer = local_dice_score_integer
                iou_score = local_iou_score
                iou_score_integer = local_iou_score_integer
            else:
                dice_score = dice_score + local_dice_score
                dice_score_integer = dice_score_integer + local_dice_score_integer
                iou_score = iou_score + local_iou_score
                iou_score_integer = iou_score_integer + local_iou_score_integer

            initial_label = y

            if batch == 0:
                true_label = initial_label
            else:
                true_label = torch.cat((true_label, y), 0)


        return true_label, total_loss, dice_score, dice_score_integer, iou_score, iou_score_integer


    # Define test loop -----------------------------------------------------------------------------------------------------

    def test_loop(dataloader, model, loss_fn_1, loss_fn_2):

        size = len(dataloader.dataset)
        total_loss = 0
        counter = 0
        current = 0

        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):

                X = X.to(device)
                y = y.to(device)

                pred = torch.sigmoid(model(X))

                display = pred.cpu().detach().numpy()  # detach convert from require grad to not require grad
                # print(np.amax(display))
                # print(np.amin(display))

                display = display >= 0.5


                image = Image.fromarray(display[0, 0, :, :])
                drive_path = ('drive path')
                path = os.path.join(drive_path, 'fold_{}_image_{}.png'.format(k+1, counter))
                image.save(path)

                initial_pred = pred

                loss = loss_fn_1(pred, y) + loss_fn_2.forward(y, pred)


                loss, current = loss.item(), current + len(X)


                total_loss = total_loss + loss

                local_dice_score = performance_metric.DiceLoss()
                local_dice_score = local_dice_score.forward(pred, y)
                # print(f'local_dice_score:{local_dice_score}')

                local_dice_score_integer = performance_metric.DiceLoss_integer()
                local_dice_score_integer = local_dice_score_integer.forward(pred, y)
                # print(f'local_dice_score_integer:{local_dice_score_integer}')

                local_iou_score = performance_metric.IoULoss()
                local_iou_score = local_iou_score.forward(pred, y)
                # print(f'local_iou_score:{local_iou_score}')

                local_iou_score_integer = performance_metric.IoULoss_integer()
                local_iou_score_integer = local_iou_score_integer.forward(pred, y)
                # print(f'local_iou_score_integer:{local_iou_score_integer}')

                if batch == 0:
                    dice_score = local_dice_score
                    dice_score_integer = local_dice_score_integer
                    iou_score = local_iou_score
                    iou_score_integer = local_iou_score_integer
                else:
                    dice_score = dice_score + local_dice_score
                    dice_score_integer = dice_score_integer + local_dice_score_integer
                    iou_score = iou_score + local_iou_score
                    iou_score_integer = iou_score_integer + local_iou_score_integer


                if batch == 0:
                    prediction = initial_pred
                else:
                    prediction = torch.cat((prediction, pred), 0)


                del X, y

                counter = counter + 1

        return prediction, total_loss, dice_score, dice_score_integer, iou_score, iou_score_integer


    # start to train -------------------------------------------------------------------------------------------------------

    if torch.cuda.is_available():
        project_path = 'file path'
    else:
        project_path = 'file path'

    training_loss = []
    training_dice = []
    validation_loss = []
    validation_dice = []


    for t in range(epochs):


        print(f'k = {k+1}, Epoch {t + 1}\n train loop---------------------------- ')
        model.train()
        true_label, total_loss, dice_score, dice_score_integer, iou_score, iou_score_integer = train_loop(train_dataloader, model, loss_BCE, loss_DICE, optimizer)

        training_loss.append(total_loss/(80/train_batch_size))
        # print(type(dice_score))
        dice_score_np = dice_score.cpu().detach().numpy()
        # print(type(dice_score))
        training_dice.append(dice_score_np/(80/train_batch_size))

        # path = os.path.join(project_path, '{}_{}_{}_{}_{}_{}.pth'.format(batch_size, learning_rate, 'adaptive', gamma, (t+1), 'aug.'))
        # torch.save(model.state_dict(), path)

        print(f'k = {k+1}, Epoch {t+1}\n test loop---------------------------- ')
        model.eval()
        prediction, total_loss, dice_score, dice_score_integer, iou_score, iou_score_integer = test_loop(test_dataloader, model, loss_BCE, loss_DICE)

        validation_loss.append(total_loss/(20/test_batch_size))
        dice_score_np = dice_score.cpu().detach().numpy()
        validation_dice.append(dice_score_np/(20/test_batch_size))

        # print(type(prediction))  # tensor [5, 1, 572, 572]
        # print(len(true_label))  # tensor [5, 1, 572, 572]


        dice_score = dice_score / len(test_dataloader.dataset)
        print(f'dice score: {dice_score}')

        dice_score_integer = dice_score_integer / len(test_dataloader.dataset)
        print(f'dice_score_integer:{dice_score_integer}')

        iou_score = iou_score / len(test_dataloader.dataset)
        print(f'iou_score:{iou_score}')

        iou_score_integer = iou_score_integer / len(test_dataloader.dataset)
        print(f'iou_score_integer:{iou_score_integer}')

        scheduler.step()


    plt.plot(range(epochs), training_loss, label='training_loss')
    plt.plot(range(epochs), validation_loss, label='validation_loss')
    plt.title('training loss vs. validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    plt.plot(range(epochs), training_dice, label='training_dice')
    plt.plot(range(epochs), validation_dice, label='validation_dice')
    plt.title('training dice vs. validation dice')
    plt.xlabel('epoch')
    plt.ylabel('dice')
    plt.legend()
    plt.show()

    overall_dice_score.append(dice_score)
    overall_dice_score_integer.append(dice_score_integer)
    overall_iou_score.append(iou_score)
    overall_iou_score_integer.append(iou_score_integer)

    print(overall_dice_score)
    print(overall_dice_score_integer)
    print(overall_iou_score)
    print(overall_iou_score_integer)

    print(f'k={k+1},Done!')

print(sum(overall_dice_score)/len(overall_dice_score))
print(sum(overall_dice_score_integer)/len(overall_dice_score_integer))
print(sum(overall_iou_score)/len(overall_iou_score))
print(sum(overall_iou_score_integer)/len(overall_iou_score_integer))

print('All Done!')
