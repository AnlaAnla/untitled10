import pandas as pd
import numpy as np
import cv2
import os

path = r"C:\Users\wow38\Documents\WeChat Files\wxid_nilj9ybra13v22\FileStorage\File\2024-07\宝可梦0725.csv"

df = pd.read_csv(path)
img = np.array(df)

# img = img[:-1, :] - img[1:, :]
# img = img[:, -1:] - img[:, 1:]

img = np.clip(img, 1.8, 2)
img = img - img.min()

img = (img - img.min()) / (img.max() - img.min()) * 255
# img = cv2.flip(img, 1)

cv2.imwrite(f"img.jpg", img)
