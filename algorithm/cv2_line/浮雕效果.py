import cv2
import numpy as np

# 加载图像
img = cv2.imread(r"C:\Code\ML\Image\Card_test\test03\458a7303-e33a-4e3b-8626-156ea3269e6b_FRONT_MAIN.jpg",
                 cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (800, 1000))

# 定义卷积核
# kernel = np.array([[-1, 0, 1],
#                    [0, 0, 0],
#                    [1, 0, -1]])
kernel = np.array([[-1, 0],
                   [0, 1]])

# 执行卷积操作
result = cv2.filter2D(img, -1, kernel) + 150
result.clip(0, 255)

cv2.imwrite('temp.jpg', result)

# 显示结果
# cv2.imshow('Result', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
