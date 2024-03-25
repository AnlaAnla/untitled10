import cv2
import numpy as np


def rotateImage(image, angle):
    """
    旋转图像
    :param image: 输入图像
    :param angle: 旋转角度(角度制)
    :return: 旋转后的图像
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def find_card_contour(image):
    """
    找到卡片的轮廓
    :param image: 输入图像
    :return: 卡片的轮廓
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大的轮廓(假设卡片是最大的轮廓)
    card_contour = max(contours, key=cv2.contourArea)
    return card_contour


def correct_card_angle(image):
    """
    纠正卡片的角度
    :param image: 输入图像
    :return: 角度纠正后的图像
    """
    card_contour = find_card_contour(image)

    # 找到卡片的最小外接矩形
    rect = cv2.minAreaRect(card_contour)
    box = cv2.boxPoints(rect)

    # 计算旋转角度
    angle = rect[2]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # 旋转图像
    rotated = rotateImage(image, angle)

    return rotated


# 使用示例
image = cv2.imread(r"C:\Code\ML\Image\angle_data\test\img\2024_03_22___03_40_55.jpg")
corrected_image = correct_card_angle(image)
cv2.imwrite('temp.jpg', corrected_image)
# cv2.imshow('Corrected Card', corrected_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()