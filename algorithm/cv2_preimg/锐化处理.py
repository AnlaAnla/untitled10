import cv2
import numpy as np

# 加载图像
img = cv2.imread(r"C:\Code\ML\Image\Card_test\test03\458a7303-e33a-4e3b-8626-156ea3269e6b_FRONT_MAIN.jpg")
img = cv2.resize(img, (800, 1000))

# 使用卷积核进行锐化=============================
# 创建卷积核
# kernel = np.array([[0, -1, 0],
#                    [-1, 5, -1],
#                    [0, -1, 0]])
#
# # 锐化滤波
# sharpened = cv2.filter2D(img, -1, kernel)
# ==========================================


# 使用拉普拉斯算法进行锐化==================
sharpened = cv2.Laplacian(img, cv2.CV_64F)
# ==============================================


# 显示结果
cv2.imshow('Sharpened', sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()