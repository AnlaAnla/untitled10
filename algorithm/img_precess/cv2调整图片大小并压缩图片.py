# coding=utf-8

import cv2 as cv
import os


# 路径不可有中文,size_threshold: 长或宽大于该值，压缩图片，否之跳过
def compress_img(src_path, dst_path, size_threshold=900):
    # 当前目录读取一张图片（499Kb，1920*1080）
    img = cv.imread(src_path)

    if img.shape[0] > size_threshold and img.shape[1] > size_threshold:
        # 调整长宽（长宽各调整为原来的一半，即 0.5）
        img = cv.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)

        # 压缩图片（226Kb）
        cv.imwrite(dst_path, img, [cv.IMWRITE_JPEG_QUALITY, 80])
        cv.imwrite(dst_path, img)
        print(dst_path)


father_dir = r'C:\Code\ML\Image\card_cls\train_data6_224\train'

for dir_name in os.listdir(father_dir):
    for img_dir_name in os.listdir(os.path.join(father_dir, dir_name)):
        img_path = os.path.join(father_dir, dir_name, img_dir_name)
        compress_img(img_path, img_path)
print('end')