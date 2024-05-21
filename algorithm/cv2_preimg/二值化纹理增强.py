import cv2
import numpy as np

# 加载原始图像
img = cv2.imread(r"C:\Code\ML\Image\Card_test\test03\458a7303-e33a-4e3b-8626-156ea3269e6b_FRONT_MAIN.jpg", 0)  # 读取为灰度图像

# 计算距离变换
dist = cv2.distanceTransform(img, cv2.DIST_L2, 3)

# 归一化距离结果
norm_dist = cv2.normalize(dist, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# 增强纹理效果
enhanced_dist = (norm_dist * 100).astype(np.uint8)

# 阈值处理
_, binary = cv2.threshold(enhanced_dist, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# 叠加原始图像和二值化图像
result = cv2.bitwise_and(img, img, mask=binary)
result = cv2.resize(result, (800, 1000))
# 显示结果
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()