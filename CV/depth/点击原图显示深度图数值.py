import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
import PIL.Image as Image


def get_lightimg_4csv(csv_path):
    df = pd.read_csv(csv_path)
    img = np.array(df)
    # img = img[:-1, :] - img[1:, :]
    # img = img[:, -1:] - img[:, 1:]
    img = np.clip(img, 1.5, 2.5)

    img = img - img.min()

    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = cv2.flip(img, 1)
    return img

def get_depthimg_4csv(csv_path):
    df = pd.read_csv(csv_path)
    img = np.array(df)
    return img


# 读取深度图像
light_img = get_lightimg_4csv(
    r"C:\Users\wow38\Documents\WeChat Files\wxid_nilj9ybra13v22\FileStorage\File\2024-07\宝可梦0725.csv"
)
depth_image = get_depthimg_4csv(
    r"C:\Users\wow38\Documents\WeChat Files\wxid_nilj9ybra13v22\FileStorage\File\2024-07\宝可梦0725.csv"
)

# 创建一个matplotlib图像
fig, ax = plt.subplots(figsize=(10, 6))

# 显示深度图像
ax.imshow(light_img, cmap='gray')


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
