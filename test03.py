import torch
from PIL import Image, ImageOps, ImageFile
import numpy as np
import os
import glob
import time
from multiprocessing.pool import ThreadPool

ImageFile.LOAD_TRUNCATED_IMAGES = True

yolo_model = torch.hub.load(r"C:\Users\Administrator\.cache\torch\hub\ultralytics_yolov5_master",
                            'custom', path=r"D:\Code\ML\item2\towhee_test\yolov5s.pt", source='local')

source_dir = r"D:\BaiduNetdiskDownload\Mosaic(2019, 2021)\Mosaic21-22 Base(FansM)"

num = 0


def yolo_detect(img):

    results = yolo_model(img)

    pred = results.pred[0].cpu().numpy()
    # 这是第五列等于0的行，0为person，也就是截出人的图片
    pred = pred[pred[:, 5] == 0][:, :4]
    boxes = pred.astype(np.int32)

    max_img = get_object(img, boxes)

    return max_img


def get_object(img, boxes):
    if isinstance(img, str):
        img = Image.open(img)

    if len(boxes) == 0:
        return img

    max_area = 0

    # 选出最大的框
    x1, y1, x2, y2 = 0, 0, 0, 0
    for box in boxes:
        temp_x1, temp_y1, temp_x2, temp_y2 = box
        area = (temp_x2 - temp_x1) * (temp_y2 - temp_y1)
        if area > max_area:
            max_area = area
            x1, y1, x2, y2 = temp_x1, temp_y1, temp_x2, temp_y2

    max_img = img.crop((x1, y1, x2, y2))
    return max_img


def yolo_crop_img(img_path):
    global num
    num += 1
    print("{}, {}".format(num, img_path))

    return yolo_detect(img_path)


if __name__ == '__main__':
    img_path = r"D:\Code\ML\images\Mywork3\card_dataset_yolo\optic2019-20_10 Lou Williams\17f3289b31c35c31027386e3bad8fcf7.jpg"

    t1 = time.time()
    max_img = yolo_crop_img(img_path)

    t2 = time.time()
    print('end, time:', (t2 - t1))
