import os

from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Load a model
model = YOLO(r"C:\Code\ML\Model\Card_cls\yolo_card_seg01.pt", task='segment')  # load an official model


# model = YOLO('path/to/best.pt')  # load a custom model

# Predict with the model
# def correct(img, mask):
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
#     angle = rect[2]
#     rows, cols = mask.shape[:2]
#     if rect[1][0] > rect[1][1]:
#         angle += 90
#
#     print(f"Angle to rotate: {angle}")
#     M = cv2.getRotationMatrix2D(rect[0], angle, 1)
#     rotated = cv2.warpAffine(img, M, (cols, rows))
#
#     rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
#     return rotated
def get_angle(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    angle = rect[2]

    if rect[1][0] > rect[1][1]:
        angle += 90
    return angle

def correct(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    gray = cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=2)
    _, gray = cv2.threshold(gray, 15, 255, cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # 计算倾斜角度
    rect = cv2.minAreaRect(largest_contour)

    angle = get_angle(mask)
    rows, cols = img.shape[:2]  # 使用原始图像的尺寸

    print(f"Angle to rotate: {angle}")
    M = cv2.getRotationMatrix2D(rect[0], angle, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows))  # 使用原始图像尺寸进行旋转

    # # 找到旋转后的边界框
    # pts = cv2.boxPoints(rect)
    # pts = np.int0(cv2.transform(np.array([pts]), M))[0]
    # x, y, w, h = cv2.boundingRect(pts)
    #
    # # 裁剪出卡片区域
    # rotated = rotated[y:y+h, x:x+w]

    return rotated

def img_angle_correct(img_path):
    results = model.predict(img_path, verbose=False, max_det=4)  # predict on an image
    img = results[0].orig_img
    if results[0].masks == None:
        print(img_path, '\tNo masks detected')
        return

    mask = results[0].masks[0].chunk.numpy()[0].astype(np.uint8)
    plt.imshow(mask)
    plt.show()
    mask_corrected = correct(img, mask)

    cv2.imwrite(img_path, mask_corrected)
    print(img_path)


if __name__ == '__main__':
    img_dir_path = r"C:\Code\ML\Image\angle_data\correct\img"
    for img_name in os.listdir(img_dir_path):
        img_path = os.path.join(img_dir_path, img_name)
        img_angle_correct(img_path)

        print('_'*50)
