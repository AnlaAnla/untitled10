import os
import json
import cv2
from multiprocessing.pool import ThreadPool

def compress_img(src_path, dst_path):
    # 当前目录读取一张图片（499Kb，1920*1080）
    img = cv2.imread(src_path)

    # 调整长宽（长宽各调整为原来的一半，即 0.5）
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    # 压缩图片（226Kb）
    cv2.imwrite(dst_path, img, [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == '__main__':

    dir_path = r"D:\Code\ML\images\Mywork3\card_database\prizm\21-22"

    pool = ThreadPool(processes=6)
    for name in os.listdir(dir_path):
        file_list = os.listdir(os.path.join(dir_path, name))
        img_names = file_list[:-1]

        for img_name in img_names:
            src_path = os.path.join(dir_path, name, img_name)
            dst_path = os.path.join(dir_path, name, img_name)
            # pool.apply(func=compress_img, args=(src_path, dst_path))
            pool.apply_async(func=compress_img, args=(src_path, dst_path))
            print(img_name)
    print('end')
