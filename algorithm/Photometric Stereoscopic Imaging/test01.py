import random

import cv2
import glob
import re
import os
import numpy as np

def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def imread(file_name, intensities=None, flag=-1, scale=1.0):
    img_names = glob.glob(file_name)
    img_list = []
    if len(img_names) == 0:
        print("[No {} images]".format(file_name))
        exit()
    img_names = sorted(img_names, key=numerical_sort)
    for i, path in enumerate(img_names):
        img = cv2.resize(cv2.imread(path, flag), None, fx=scale, fy=scale)
        if  np.any(intensities):
            img = img / intensities[i]
        img_list.append(img)
    return img_list

from numpy import loadtxt

dataPath = "data/"
object = []
for obj  in (os.listdir(dataPath)):
    object.append(obj)

L_direction = loadtxt(f"data/{object[0]}/info/light_directions.txt")
intensities = loadtxt(f"data/{object[0]}/info/light_intensities.txt")
path_img = "data/bearPNG"

imgs = imread(path_img + f"/{object[0]}/*", intensities=intensities)
mask = imread(f"data/{object[0]}/info/mask.png")[0]