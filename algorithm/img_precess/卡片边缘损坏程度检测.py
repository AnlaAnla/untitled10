import cv2
import numpy as np

# 加载图像
img = cv2.imread(r"C:\Code\ML\Image\Card_test\test\1234aa.jpg")
img = cv2.resize(img,(1000,1000))

def addedge(img):
    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 高斯滤波降噪
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 计算图像梯度
    kernel = np.array([[-1, -1, 0],
                       [-1, 0, 1],
                       [0, 1, 1]])
    gradient = cv2.filter2D(gray, -1, kernel)

    # 提取边缘
    edges = cv2.Canny(gradient, 100, 200)

    # 膨胀操作使划痕更明显
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # 在原始图像上叠加处理过的边缘
    result = cv2.addWeighted(img, 0.6, cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR), 0.3, 0)
    return result
# 浮雕效果
def relief(img):
    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 计算图像梯度
    kernel = np.array([[-1, -1, 0],
                       [-1, 0, 1],
                       [0, 1, 1]])
    gradient = cv2.filter2D(gray, -1, kernel)

    # 将梯度值缩放到 0-255 范围
    gradient = cv2.normalize(gradient, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 将梯度值映射到灰度图像
    relief = cv2.bitwise_not(gradient)

    # 模糊处理浮雕图像
    # blur = cv2.GaussianBlur(relief, (5, 5), 0)
    return relief


result = cv2.addWeighted(img, 0.2, cv2.cvtColor(relief(img), cv2.COLOR_GRAY2BGR), 0.8, 1)
# 显示结果
cv2.imshow('Relief', result)
cv2.waitKey(0)
cv2.destroyAllWindows()