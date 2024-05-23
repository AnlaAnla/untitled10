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



# 加载图像
img = cv2.imread(r"C:\Code\ML\Image\Card_test\test03\458a7303-e33a-4e3b-8626-156ea3269e6b_FRONT_MAIN.jpg")
img = cv2.resize(img, (800, 1000))




# 显示结果
cv2.imshow('img', img)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
