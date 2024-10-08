import cv2
import numpy as np

# 1. 读取图像
image = cv2.imread(r"C:\Code\ML\Image\Card_test\test03\2024_06_20___02_46_36.jpg")

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 高斯模糊去噪
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 应用边缘检测
edges = cv2.Canny(blurred, 50, 150)

cv2.imshow("edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()