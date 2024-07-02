import threading
import cv2
from queue import Queue
from functools import partial
import os
import glob


def compress_img(src_path, ratio=0.5):
    # 当前目录读取一张图片
    img = cv2.imread(src_path)
    if img.shape[0] < 1200 or img.shape[1] < 1200:
        img = None
        return
    # 调整长宽
    img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)

    cv2.imwrite(src_path, img)

    # 释放内存
    img = None


def worker(image_paths, progress_queue, compress_func):
    for src_path in image_paths:
        compress_func(src_path)
        progress_queue.put(1)


if __name__ == '__main__':

    image_paths = glob.glob(r"C:\Code\ML\Image\yolo_data02\Caed_Pokemon_box\train\*.jpg")

    # 创建一个队列来接收处理进度
    progress_queue = Queue()

    # 创建线程
    num_threads = 4
    compress_partial = partial(compress_img, ratio=0.3)
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
