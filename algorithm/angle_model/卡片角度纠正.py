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


def correct_angle(img_path, save_dir=r'C:\Code\ML\Image\angle_data\plot'):
    # 读取图像
    img = cv2.imread(img_path)

    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    gray = cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=2)
    gray = imgradient(gray, 3).astype(np.uint8)

    _, gray = cv2.threshold(gray, 15, 255, cv2.THRESH_OTSU)
    gray = gray.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    # 使用自适应阈值进行预处理
    # gray = cv2.GaussianBlur(gray, (7, 7), 1)

    # 应用Canny边缘检测
    # edges = cv2.Canny(gray, 100, 200)

    # 寻找轮廓
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # 计算倾斜角度
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect).astype(int)
    angle = rect[2]

    P1 = np.array(box[0])
    P2 = np.array(box[3])

    if P2[0] > P1[0]:
        angle = -angle

    # 模型判断
    img_pil = transform_image(img_path)
    y = model(img_pil).item()

    # 使用模型判断的角度矫正
    # if y > 90:
    #     angle = -angle
    img_name = os.path.split(img_path)[-1]
    print(img_name, ' \tangle:', angle, '\tpredict:', y)

    # 旋转矫正
    rows, cols = img.shape[:2]

    if abs(angle) > 70:
        angle = abs(angle) - 90
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle/2, 1)
    corrected = cv2.warpAffine(img, M, (cols, rows))

    angle2 = 90 - y
    # if angle2 < 10:
    #     angle2 = 0
    M2 = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle2/2, 1)
    corrected2 = cv2.warpAffine(img, M2, (cols, rows))

    # 显示
    plt.figure(figsize=(15, 10))
    plt.axis('off')

    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGBA))
    plt.title('img')

    plt.subplot(132)
    plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
    plt.title('opencv')

    plt.subplot(133)
    plt.imshow(cv2.cvtColor(corrected2, cv2.COLOR_BGR2RGBA))
    plt.title('model angle')

    plt.savefig(os.path.join(save_dir, img_name), dpi=80, bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':

    # model = torch.load(r"C:\Code\ML\Model\angle_model04.pt")
    model = torch.jit.load(r"C:\Code\ML\Model\angle_model\script_angle_model06.pt")
    model.eval()

    # dir_path = r'C:\Code\ML\Image\angle_data\test\img'
    # for img_name in os.listdir(dir_path):
    #     img_path = os.path.join(dir_path, img_name)
    #     correct_angle(os.path.join(dir_path, img_path))
    correct_angle(r"C:\Code\ML\Image\Card_test\test02\23.jpg")

    # img = cv2.imread(p[1])
    # img = cv2.resize(img, (32, 32))
    #
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # for path in p:
    #     correct_angle(path)

    # correct_angle(r"C:\Code\ML\Image\test03\3 (8).jpg")
