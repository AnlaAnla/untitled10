#e17.1HandDrawPic.py
import cv2
from PIL import Image
import numpy as np

coef_1 = 8.
coef_2 = 8.
coef_3 = 50.

vec_el = np.pi / coef_1  # 光源的俯视角度，弧度值    #vec_el = np.pi/2.2 # 光源的俯视角度，弧度值
vec_az = np.pi / coef_2  # 光源的方位角度，弧度值     # vec_az = np.pi/4. # 光源的方位角度，弧度值
depth = coef_3  # (0-100)  # 梯度系数                       # depth = 10. # (0-100)
im = Image.open(r"C:\Code\ML\Image\Card_test\test03\458a7303-e33a-4e3b-8626-156ea3269e6b_FRONT_MAIN.jpg").convert('L')  # 转为灰度图
im = im.resize((800, 1000))

a = np.asarray(im).astype('float')
grad = np.gradient(a)  #取图像灰度的梯度值
grad_x, grad_y = grad  #分别取横纵图像梯度值
grad_x = grad_x * depth / 100.
grad_y = grad_y * depth / 100.
dx = np.cos(vec_el) * np.cos(vec_az)  #光源对x 轴的影响
dy = np.cos(vec_el) * np.sin(vec_az)  #光源对y 轴的影响
dz = np.sin(vec_el)  #光源对z 轴的影响
A = np.sqrt(grad_x ** 2 + grad_y ** 2 + 1.)
uni_x = grad_x / A
uni_y = grad_y / A
uni_z = 1. / A
a2 = 255 * (dx * uni_x + dy * uni_y + dz * uni_z)  #光源归一化
a2 = a2.clip(0, 255)

cv2.imshow('A', a2)
cv2.waitKey(0)

cv2.destroyAllWindows()
# im2 = Image.fromarray(a2.astype('uint8'))  #重构图像
# savePath = '#coef_1#' + str(coef_1) + '#coef_2#' + str(coef_2) + '#coef_3#' + str(coef_3) + '.jpg'
# im2.save(savePath)
