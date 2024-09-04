import cv2
import numpy as np

# 读入原始图像
img = cv2.imread(r"C:\Code\ML\Image\Card_test\test03\11.jpg")

# 预处理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(gray, 10, 150)

# 连通区域分析
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)
sizes = stats[1:, -1]
max_index = np.argmax(sizes)
card_area = np.where(output == max_index + 1, 255, 0).astype('uint8')

# 获取最小外接矩形
x, y, w, h = cv2.boundingRect(card_area)
card_rect = img[y:y+h, x:x+w]

# 透视变换并输出结果
pts1 = np.float32([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(img, matrix, (w, h))

cv2.imshow('thresh', edges)
cv2.imshow('card', result)
cv2.waitKey(0)
cv2.destroyAllWindows()