import cv2
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
import PIL.Image as Image

# 读取深度图像
depth_image = cv2.imread(r"C:\Users\wow38\Pictures\Screenshots\map1.png", cv2.IMREAD_ANYDEPTH)

# 创建一个matplotlib图像
fig, ax = plt.subplots(figsize=(10, 6))

# 显示深度图像
ax.imshow(depth_image, cmap='gray')

# 定义一个函数,在鼠标点击时显示该位置的数值
def on_click(event):
    if event.inaxes == ax:
        x, y = int(event.xdata), int(event.ydata)
        value = depth_image[y, x]
        ax.text(x, y, str(value), color='red', fontsize=10, ha='center', va='center')
        fig.canvas.draw()

# 连接鼠标点击事件
fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()