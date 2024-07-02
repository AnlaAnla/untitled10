import cv2
import numpy as np


def equalize_colorimg(img):
    b, g, r = cv2.split(img)

    # 对每个通道进行直方图均衡化
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)

    # 合并三个均衡化后的通道
    result = cv2.merge((b_eq, g_eq, r_eq))
    return result


img = cv2.imread(r"C:\Code\ML\Image\Card_test\test03\2024_06_20___02_46_36.jpg", 0)
img = cv2.resize(img, (800, 1000))
# img = img + 100
# img = img.clip(0, 255)

# 锐化处理
# 创建卷积核 ===============================
# kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
# img = cv2.filter2D(img, -1, kernel=kernel)

# 直方图均衡化增强对比度
# img = equalize_colorimg(img)

# img = cv2.Laplacian(img, cv2)
# HPF高通滤波器
# GBlur = cv2.GaussianBlur(img, (11, 11), 0)
# img = img-GBlur
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# img = cv2.bilateralFilter(img, 9, 75, 75)
# 锐化滤波
# ========================================

# result = img[:, :-1] - img[:, 1:] + 150

result = img[:-1, :] - img[1:, :]
# 创建一个布尔掩码数组,用于标记需要加200和加100的像素位置
mask_add_200 = np.abs(result) < 40
mask_add_100 = np.abs(result) >= 40

# 创建两个值数组,分别对应需要加200和加100的值
values_add_200 = np.full_like(result, 200)
values_add_100 = np.full_like(result, 150)

# 使用布尔掩码和值数组进行运算
result = np.where(mask_add_200, result + values_add_200, result)
result = np.where(mask_add_100, result + values_add_100, result)

result = result.clip(0, 255)
cv2.imshow('img', img)
cv2.imshow('result', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
