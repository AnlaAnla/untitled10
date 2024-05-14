import numpy as np
from transformers import pipeline
from PIL import Image
import cv2
import matplotlib.pyplot as plt

pipe = pipeline(task="depth-estimation", model=r"C:\Code\ML\Model\huggingface\depth-anything-small-hf")
img = Image.open(r"C:\Users\wow38\Pictures\Screenshots\map2.png")
depth = pipe(img)['depth']
depth_image = np.asarray(depth)

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
