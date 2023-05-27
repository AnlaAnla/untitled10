import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Administrator\Downloads\2020061322284248.bmp", 0)
watermark = cv2.imread(r"C:\Users\Administrator\Downloads\20200613222957926.bmp", 0)

# 因为水印图像就是让人不易察觉也不影响原图像，所以要将水印非0位全部替换位最小值1
w = watermark[:, :] > 0
watermark[w] = 1
# 嵌入水印
r, c = img.shape
# 生成元素都是254的数组
img254 = np.ones((r, c), dtype=np.uint8) * 254
# 获取高7位平面
imgH7 = cv2.bitwise_and(img, img254)

# 将水印嵌入即可
water_img = cv2.bitwise_or(imgH7, watermark)
cv2.imshow("1", img)
cv2.imshow("2", watermark * 255)
cv2.imshow("3", water_img)
# 生成都是1的数组
img1 = np.ones((r, c), dtype=np.uint8)
# 提取水印
water_extract = cv2.bitwise_and(water_img, img1)

# 将水印里面的1还原成255
w = water_extract[:, :] > 0
water_extract[w] = 255
cv2.imshow("4", water_extract)

cv2.waitKey()
cv2.destroyAllWindows()
