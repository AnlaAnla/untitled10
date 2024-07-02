import time
import numpy as np
import cv2

# cap = cv2.VideoCapture(0)
#
# num = 0
# while cap.isOpened():
#     num += 1
#     print(num)
#     ret, frame = cap.read()
#     if ret == True:
#         if num == 10:
#             cv2.imwrite('1.bmp', frame)
#         if num == 50:
#             cv2.imwrite('2.bmp', frame)
#             break
#
#         # cv2.imshow('frame', frame)
#         # cv2.waitKey(0)
#
#
#         time.sleep(0.1)
#
# img1 = cv2.imread('1.bmp')
# img2 = cv2.imread('2.bmp')

img3 = img1 - img2

cv2.imshow('img3', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
