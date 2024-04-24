import cv2
import numpy as np

color_red = (0, 0, 255)  # 画笔颜色(BGR)


img = cv2.imread(r"C:\Code\ML\Image\Card_test\test\ada (1).jpg")
img = cv2.resize(img, (512, 512))
def draw_counter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转化为灰度图
    cv2.imshow('gray', gray)

    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # 阈值二值化

    cv2.imshow('thresh', thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
    largest_contour = max(contours, key=cv2.contourArea)

    rect = cv2.minAreaRect(largest_contour)  # 由于该图像只有一个轮廓，所以直接取 contours[0]
    print(rect)
    box = cv2.boxPoints(rect).astype(int)
    angle = rect[2]

    cv2.drawContours(img, [box], 0, color_red, 2)

    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle / 2, 1)
    corrected = cv2.warpAffine(img, M, (cols, rows))
    return corrected



kernel = np.ones((10, 10), dtype=np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
opening = draw_counter(opening)

image_stack = np.hstack((img, opening))



# cv2.imshow('origin', img)
cv2.imshow('result', image_stack)

cv2.waitKey(0)
cv2.destroyAllWindows()
