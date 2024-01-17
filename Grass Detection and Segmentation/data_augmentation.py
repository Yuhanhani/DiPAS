from torchvision import transforms
import numpy
import torch
import albumentations as A

class data_augmentation_transform():

    def __init__(self, phase):
        self.phase = phase

    def transform(self):

         transform_dict = {

            'train': A.Compose([
                # transforms.ToPILImage(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 

                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
                A.Resize(height=1000, width=1000),
                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

            ]),


            'test': A.Compose([
                # transforms.ToPILImage(),
                A.Resize(height=1000, width=1000),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                # transforms.Normalize([105.424, 105.424, 105.424], [68.886, 68.886, 68.886])
         ]),

    }

         return transform_dict[self.phase]
