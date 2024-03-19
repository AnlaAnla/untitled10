import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
from AngleNet import AngleNet

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def transform_image(img_path):
    img = Image.open(img_path).convert('RGB')
    return transform(img).unsqueeze(0)


if __name__ == '__main__':

    # model = AngleNet()
    # model.load_state_dict(torch.load('angle_model03.pth'))
    model = torch.load('angle_model04.pt')
    model.eval()

    # print(model)
    dir_path = r"C:\Code\ML\Image\angle_data\train\img"
    for img_name in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_name)
        img = transform_image(img_path)

        # a = img[0].transpose(0, 2)
        # plt.imshow(a)
        # plt.show()

        y = model(img)
        print(img_name, ': ', y.item())
