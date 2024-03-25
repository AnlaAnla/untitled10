import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread(r"D:\Code\ML\images\work1\1.jpg", 0)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
template = cv2.imread(r"D:\Code\ML\images\test\trial\temp02.jpg", 0)
h, w = template.shape[:2]  # rows->h, cols->w

# 2.标准相关模板匹配
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8

# 3.这边是Python/Numpy的知识，后面解释
loc = np.where(res >= threshold)  # 匹配程度大于%80的坐标y,x
for pt in zip(*loc[::-1]):  # *号表示可选参数
    right_bottom = (pt[0] + w, pt[1] + h)
    cv2.rectangle(img, pt, right_bottom, (0, 0, 255), 2)


cv2.imshow("MatchResult----MatchingValue=", img)
cv2.waitKey()
cv2.destroyAllWindows()