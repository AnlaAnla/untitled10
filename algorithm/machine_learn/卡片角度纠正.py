import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


def correct_angle(img_path):
    # 读取图像
    img = cv2.imread(img_path)

    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 使用自适应阈值进行预处理
    # gray = cv2.GaussianBlur(gray, (7, 7), 1)

    # 应用Canny边缘检测
    edges = cv2.Canny(gray, 100, 200)

    # 寻找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)


    # 计算倾斜角度
    rect = cv2.minAreaRect(largest_contour)
    print(os.path.split(img_path)[-1], rect)
    angle = rect[2]


    cv2.circle(img, center=(int(rect[0][0]), int(rect[0][1])), radius=20, color=(0, 255, 0), thickness=-1)
    cv2.arrowedLine(img, (int(rect[0][0]), int(rect[0][1])),
                    (int(rect[0][0] - 5 * rect[1][0]), int(rect[0][1] - 5 * rect[1][1])),
                    (0, 0, 255), 20, 0, 0, 0.2)

    # 旋转矫正
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle / 2, 1)
    corrected = cv2.warpAffine(img, M, (cols, rows))

    # 显示
    plt.figure(figsize=(15, 10))
    plt.axis('off')

    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGBA))
    plt.title('img')

    plt.subplot(132)
    plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
    plt.title('canny')

    plt.subplot(133)
    plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGBA))
    plt.title('corrected')
    plt.show()


def get_rect(img_path):
    img = cv2.imread(img_path)

    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 应用Canny边缘检测
    edges = cv2.Canny(gray, 100, 200)

    # 寻找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    # cv2.drawContours(img, contours, 0, (0, 255, 255), 3)

    # 计算倾斜角度
    rect = cv2.minAreaRect(largest_contour)
    return rect


if __name__ == '__main__':
    dir_path = r'C:\Code\ML\Image\angle_data\train\img_r'
    for img_name in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_name)
        correct_angle(os.path.join(dir_path, img_path))

    # img = cv2.imread(p[1])
    # img = cv2.resize(img, (32, 32))
    #
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # for path in p:
    #     correct_angle(path)

    # correct_angle(r"C:\Code\ML\Image\test03\3 (8).jpg")
