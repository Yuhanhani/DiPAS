
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




class CustomImageDataset(Dataset):

    def __init__(self, img_path, mask_path, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.transform = transform
        self.images = sorted(os.listdir(img_path))


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_dir = os.path.join(self.img_path, self.images[index])
        mask_dir = os.path.join(self.mask_path, self.images[index].replace('.jpg', '.png'))
        print(mask_dir)
        image = np.array(Image.open(img_dir))
        mask = np.array(Image.open(mask_dir))


        if self.transform:
            augmentations = self.transform.transform()(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
            image = np.transpose(image, (2, 0, 1))
            mask = np.expand_dims(mask, axis=0)

        image = torch.FloatTensor(image)
        mask = torch.FloatTensor(mask)


        return image, mask
