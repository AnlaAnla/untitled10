import cv2
import numpy as np
import random


img = cv2.imread(r"C:\Code\ML\Image\yolo_data02\Card_scratch\pokemon_not_scratch\image (1).jpg")
mask = cv2.imread(r"C:\Code\ML\Image\yolo_data02\Card_scratch\scratch_mask\0ab629c243cd4eadbde7642e2f6f5a8a.png")

img_copy = img.copy()
img[np.where((mask != [255, 255, 255, 0]).all(axis=2))] = 0
# 获取Alpha通道
alpha_channel = mask[:, :, 3]
# 将Alpha值为0的区域对应的RGB值设置为0
mask[alpha_channel == 0] = [0, 0, 0, 0]
mask = mask[:, :, :3]

result = img + mask

cv2.imshow("img", img_copy)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
