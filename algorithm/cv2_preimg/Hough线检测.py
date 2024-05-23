import cv2
import numpy as np

# 读取图像
img = cv2.imread(r"C:\Code\ML\Image\Card_test\test03\35465.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 高斯模糊去噪
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 边缘检测
edges = cv2.Canny(blur, 50, 150)

# Hough线检测
lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

# 绘制检测到的线段
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

# 显示结果
cv2.imshow('Card with Crack', img)
cv2.waitKey(0)
cv2.destroyAllWindows()