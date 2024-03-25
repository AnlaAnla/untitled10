import os
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def transform_image(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return transform(img).unsqueeze(0)


def imgradient(img, sobel):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel)
    return np.sqrt(np.multiply(sobelx, sobelx) + np.multiply(sobely, sobely))

# 将改正后的图片覆盖原图片
def correct_angle(img_path):
    # 读取图像
    img = cv2.imread(img_path)

    # 模型判断
    img_pil = transform_image(img_path)
    y = model(img_pil).item()

    print(os.path.split(img_path)[-1], '\tpredict:', y)

    # 旋转矫正
    rows, cols = img.shape[:2]

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90 - y, 1)
    corrected = cv2.warpAffine(img, M, (cols, rows))
    return corrected



if __name__ == '__main__':

    # model = torch.load(r"C:\Code\ML\Model\angle_model04.pt")
    model = torch.jit.load(r"C:\Code\ML\Model\angle_model\script_angle_model05.pt")
    model.eval()

    dir_path = r'C:\Code\ML\Image\angle_data\test\img'
    save_dir = r'C:\Code\ML\Image\test\no_correct'
    for img_name in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_name)
        correct_img = correct_angle(os.path.join(dir_path, img_path))

        cv2.imwrite(os.path.join(save_dir, img_name), correct_img)
