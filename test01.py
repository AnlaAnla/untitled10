import cv2
import numpy as np

img = cv2.imread(
    r"C:\Code\ML\Image\yolo_data02\Card_scratch\card_scratch\train\images\image - 2024-07-23T165027.0217f091b14dc004675a0f64df3cef410e3.jpg")
mask = cv2.imread(r"C:\Code\ML\Image\Card_test\test03\scratch\222.png", cv2.IMREAD_UNCHANGED)

img_copy = img.copy()
img[np.where((mask != [255, 255, 255, 0]).all(axis=2))] = 0
# 获取Alpha通道
alpha_channel = mask[:, :, 3]
# 将Alpha值为0的区域对应的RGB值设置为0
mask[alpha_channel == 0] = [0, 0, 0, 0]
mask = mask[:, :, :3]


result = img + mask
# # 显示结果
cv2.imshow("img", img_copy)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
