import cv2
import numpy as np

img = cv2.imread(r"C:\Code\ML\Image\Card_test\test03\2 (10).jpg", 0)
img = cv2.GaussianBlur(img, (5, 5), 0)

kernel = np.ones((5, 5), np.uint8)

# 闭运算
# img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
# 开运算,先腐蚀后膨胀
# img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

edges = cv2.Canny(img, 100, 200)
# cv2.imshow('edges', edges)
# cv2.waitKey(0)x
# cv2.destroyAllWindows()

precess = 1

if precess == 1:
    # 1.进行霍夫线变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=60, maxLineGap=10)

    # 遍历检测到的线段,过滤掉较短的线段
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if length > 0:  # 设置阈值过滤较短线段
            filtered_lines.append(line)

    # 在原图上绘制过滤后的线段
    line_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for line in filtered_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('Detected Lines', line_img)
    cv2.imshow('canny edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif precess == 2:

    # 创建矩形模板
    w, h = img.shape
    rect = np.zeros((w, h), dtype=np.uint8)
    rect[:, :] = 255
    rect[20:w - 20, 20:h - 20] = 0

    # 匹配矩形模板
    res = cv2.matchShapes(edges, rect, 1, 0.0)
    print('Match Score:', res)

    # 检测不匹配区域
    diff = cv2.absdiff(edges, rect)
    _, diff = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

    # 显示结果
    cv2.imshow('Card Edges', edges)
    cv2.imshow('Rectangle Template', rect)
    cv2.imshow('Diff', diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
