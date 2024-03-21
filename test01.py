import os
import torch
import torchvision
from torchvision import datasets, transforms
import PIL.Image as Image

import matplotlib.pyplot as plt

data_transforms = {
        'train': transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.RandAugment(),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomApply([transforms.GaussianBlur(5)], p=0.3),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # transforms.Resize((224, 224)),

            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

img = Image.open(r"C:\Code\ML\Image\card_cls\train_data4_224\train\1 Kevin Durant\6YQAAOSwfvZjbZOm.jpg")
print(img.size)