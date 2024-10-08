import cv2
import numpy as np

# 全局变量
thresh1 = 100
thresh2 = 200
edges = None


# 回调函数,用于更新Canny阈值
def update_canny(val=None):
    global edges
    edges = cv2.Canny(img, thresh1, thresh2)
    cv2.imshow('Edges', edges)


# 重置函数,用于恢复默认参数
def reset_params(val=None):
    global thresh1, thresh2
    thresh1 = 100
    thresh2 = 200
    cv2.setTrackbarPos('Threshold1', 'Controls', thresh1)
    cv2.setTrackbarPos('Threshold2', 'Controls', thresh2)
    update_canny()


# 加载图像
img = cv2.imread(r"C:\Code\ML\Image\Card_test\test03\35465.jpg", 0)

# 创建窗口和滚动条
cv2.namedWindow('Edges')
cv2.createTrackbar('Threshold1', 'Edges', thresh1, 255, update_canny)
cv2.createTrackbar('Threshold2', 'Edges', thresh2, 255, update_canny)

# 创建按钮
cv2.namedWindow('Controls')
cv2.createButton('Reset', reset_params, None, cv2.QT_PUSH_BUTTON, 1)

# 初始化
update_canny()

# 等待按键按下
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# 释放资源
cv2.destroyAllWindows()
