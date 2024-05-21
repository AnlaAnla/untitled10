import cv2
import numpy as np

img = cv2.imread(r"C:\Code\ML\Image\Card_test\test03\458a7303-e33a-4e3b-8626-156ea3269e6b_FRONT_MAIN.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (800, 1000))


# 锐化处理
# 创建卷积核 ===============================

# 锐化滤波
# ========================================

result = img[:, :-1] - img[:, 1:] + 150
result = result.clip(0, 255)
cv2.imshow('img', img)
cv2.imshow('result', result)

cv2.waitKey(0)
cv2.destroyAllWindows()