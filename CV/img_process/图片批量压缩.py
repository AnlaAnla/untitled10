import threading
import cv2
from queue import Queue
from functools import partial
import os
import numpy as np
import glob


def letterbox(src_path, target_size=1024):
    # 读取原始图像
    img = cv2.imread(src_path)
    if img.shape[0] <= target_size or img.shape[1] <= target_size:
        img = None
        return

    # 获取原始图像的高度和宽度
    h, w = img.shape[:2]

    # 计算目标缩放比例
    # target_size = 1024
    ratio = target_size / max(h, w)

    # 计算缩放后的尺寸
    new_h = int(h * ratio)
    new_w = int(w * ratio)

    # 执行缩放
    img = cv2.resize(img, (new_w, new_h))

    # 创建一个新的画布,用黑色填充
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    # 将缩放后的图像粘贴到新画布的中心
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img

    cv2.imwrite(src_path, canvas)


def compress_img(src_path, ratio=0.5):
    # 当前目录读取一张图片
    img = cv2.imread(src_path)
    if img.shape[0] < 1024 or img.shape[1] < 1024:
        img = None
        return
    # 调整长宽
    img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)

    cv2.imwrite(src_path, img)

    # 释放内存
    img = None


def resize_img(src_path, target_size=640):
    img = cv2.imread(src_path)
    if img.shape[0] <= target_size or img.shape[1] <= target_size:
        img = None
        return
    img = cv2.resize(img, (target_size, target_size))
    cv2.imwrite(src_path, img)


def worker(image_paths, progress_queue, compress_func):
    for src_path in image_paths:
        compress_func(src_path)
        progress_queue.put(1)


if __name__ == '__main__':

    image_paths = glob.glob(r"D:\Code\ML\Image\_CLASSIFY\card_cls2\train_serices_cls_data_yolo224\*\*\*")

    # 创建一个队列来接收处理进度
    progress_queue = Queue()

    # 创建线程
    num_threads = 3
    # 设置压缩方式
    compress_partial = partial(resize_img, target_size=224)
    # compress_partial = partial(compress_img, ratio=0.3)
    # compress_partial = partial(letterbox, target_size=1024)

    threads = []
    chunk_size = len(image_paths) // num_threads
    for i in range(num_threads):
        start = i * chunk_size
        end = start + chunk_size if i < num_threads - 1 else len(image_paths)
        thread = threading.Thread(target=worker, args=(image_paths[start:end], progress_queue, compress_partial))
        thread.start()
        threads.append(thread)

    # 获取总图片数量
    total_images = len(image_paths)
    processed_images = 0

    # 从队列中获取进度更新
    while processed_images < total_images:
        progress = progress_queue.get()
        processed_images += progress
        print(f"处理进程: {processed_images}/{total_images}")

    # 等待所有线程完成
    for thread in threads:
        thread.join()

    print("All images processed.")
