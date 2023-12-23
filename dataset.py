# dataset involves two transforms

import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import csv
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


# deal with image: can use np, cv2, pil, tensor

class CustomImageDataset(Dataset):  # inherent from dataset class

    def __init__(self, img_path, mask_path, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.transform = transform
        self.images = sorted(os.listdir(img_path))  # otherwise in random order (ok but for now just make them in order)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_dir = os.path.join(self.img_path, self.images[index])
        mask_dir = os.path.join(self.mask_path, self.images[index].replace('.jpg', '.png'))
        print(mask_dir)
        image = np.array(Image.open(img_dir))  # max 255, min 0, size (7024, 8192, 3)
        # mask = np.array(Image.open(mask_dir).convert('L'), dtype=np.float32)  # convert to grey scale for safety
        # mask[mask == 255.0] = 1.0  # convert to binary case
        mask = np.array(Image.open(mask_dir))  # labelled place is 1, other is 0, size (7024, 8192)


        if self.transform:
            augmentations = self.transform.transform()(image=image, mask=mask)
            image = augmentations['image']   # (572, 572, 3)
            mask = augmentations['mask']     # (572, 572)  min 0, max 1
            image = np.transpose(image, (2, 0, 1))  # (3, 572, 572)
            mask = np.expand_dims(mask, axis=0)  # (1, 572, 572)  min 0, max 1

        image = torch.FloatTensor(image)
        mask = torch.FloatTensor(mask)
        # all size & value properties unchanged

        return image, mask
