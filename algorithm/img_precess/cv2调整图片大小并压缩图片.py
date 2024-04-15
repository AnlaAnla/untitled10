# coding=utf-8

import cv2 as cv
import os
from PIL import Image


# 路径不可有中文,size_threshold: 长或宽大于该值，压缩图片，否之跳过
def compress_img(src_path, dst_path, size_threshold=900):
    # 当前目录读取一张图片（499Kb，1920*1080）
    try:
        img = Image.open(src_path).convert('RGB').resize((224, 224))
        img.save(dst_path)
    except Exception as e:
        print(src_path, e)

    # img = cv.resize(img, (224, 224))
    # cv.imwrite(dst_path, img)
    # if img.shape[0] > size_threshold and img.shape[1] > size_threshold:
    #     # 调整长宽（长宽各调整为原来的一半，即 0.5）
    #     img = cv.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
    #
    #     # 压缩图片（226Kb）
    #     cv.imwrite(dst_path, img, [cv.IMWRITE_JPEG_QUALITY, 80])
    #     cv.imwrite(dst_path, img)
    #     print(dst_path)


father_dir = r'C:\Code\ML\Image\card_cls\All_Onepiece_Card\data01'
father_save_dir = r'C:\Code\ML\Image\card_cls\train_onepiece01_224'

for i, img_name in enumerate(os.listdir(father_dir)):
    img_path = os.path.join(father_dir, img_name)

    save_dir = os.path.join(father_save_dir, os.path.splitext(img_name)[0])
    save_path = os.path.join(save_dir, img_name)
    # print(img_path, save_path)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    compress_img(img_path, save_path)
    print(i, img_name)
print('end')
