from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Load a model
model = YOLO(r"C:\Code\ML\Model\Card_cls\yolo_card_seg01.pt", task='segment')  # load an official model


# model = YOLO('path/to/best.pt')  # load a custom model

# Predict with the model
def correct(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    angle = rect[2]
    rows, cols = mask.shape[:2]
    if rect[1][0] > rect[1][1]:
        angle += 90

    print(f"Angle to rotate: {angle}")
    M = cv2.getRotationMatrix2D(rect[0], angle, 1)
    rotated = cv2.warpAffine(mask, M, (cols, rows))
    return rotated

def minRect(mask):
    # mask是你的二值图像
    points = np.argwhere(mask > 0)  # 提取mask中非零点的坐标
    points = points.reshape(-1, 2)  # 将点坐标重塑为 (N, 2) 的形状

    rectangle = cv2.minAreaRect(points)  # 计算最小外接矩形
    angle = rectangle[2]  # 提取旋转角度
    return angle


def pac(mask):
    mean = np.mean(np.argwhere(mask > 0), axis=0)
    ys, xs = np.nonzero(mask)
    points = np.array(list(zip(xs - mean[1], ys - mean[0])))
    mean, eigenvectors = cv2.PCACompute(points, mean=None)
    angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])

    if angle > -np.pi / 2:
        angle += np.pi  # 将-180到-90度的角度转为正值
    angle = angle * 180 / np.pi  # 将弧度制转换为角度制
    return angle


# 直线拟合
def fitLine(mask):
    coords = np.argwhere(mask > 0)
    vx, vy, cx, cy = cv2.fitLine(coords, cv2.DIST_L2, 0, 0.01, 0.01)
    angle = np.arctan2(vy, vx)[0]

    angle = angle * 180 / np.pi
    if angle > 90:
        angle = 180 - angle  # 将90到180度的角度转为逆时针角度
    return angle


def getAngle(img_path):
    results = model(img_path)  # predict on an image
    mask = results[0].masks[0].data.numpy()[0]

    print('rect', minRect(mask))
    print('pac', pac(mask))
    print('fitLine', fitLine(mask))

    plt.imshow(mask, cmap='gray')
    plt.show()


if __name__ == '__main__':
    getAngle(r"C:\Code\ML\Image\angle_data\train\img\right1 (8).jpg")
    print()