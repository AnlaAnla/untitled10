import asyncio
import os
import time
from multiprocessing import Pool
from pathlib import Path

import PIL.Image as Image
from tqdm import tqdm

BATCH_SIZE = 100  # 根据机器性能调整


def check_batch(img_paths_batch):
    bad_paths = []
    for img_path in img_paths_batch:
        try:
            img = Image.open(img_path)
            if img.size[0] < 224 or img.size[1] < 224:
                bad_paths.append(img_path)

        except PermissionError as e:
            if e.errno == 13:
                print(f" !!!!!! 权限错误: 无法访问路径，可能是目录或没有权限: {img_path}")
            else:
                print(f"  !!!!!!  权限错误: {img_path}, \t错误信息: {e}")
        except Exception as e:
            print(f"读取图片失败: {img_path}, \t错误信息: {e}")
            bad_paths.append(img_path)

    return bad_paths


def process_images(img_paths, num_processes):
    total_bad_paths = []
    with Pool(processes=num_processes) as pool:
        # 分批处理图片路径
        batches = [
            img_paths[i: i + BATCH_SIZE]
            for i in range(0, len(img_paths), BATCH_SIZE)
        ]
        # 使用 imap_unordered 并添加进度条
        results = pool.imap_unordered(check_batch, batches)
        for bad_paths in tqdm(results, total=len(batches)):
            total_bad_paths.extend(bad_paths)

    return total_bad_paths


if __name__ == "__main__":
    img_dir = Path(
        r"D:\Code\ML\Image\_TEST_DATA\Card_series_cls\2023-24"
    )
    img_paths = list(img_dir.glob("*/*"))  # 使用 pathlib 简化路径操作

    num_processes = os.cpu_count() or 1  # 获取 CPU 核心数

    t1 = time.time()
    bad_paths = process_images(img_paths, num_processes)
    t2 = time.time()

    # 最后统一删除损坏或尺寸不符的图片
    # for path in bad_paths:
    #     os.remove(path)
    #     print(f"已删除: {path}")

    print(
        "===========",
        len(img_paths),
        "  time：",
        (t2 - t1),
        " bad_num: ",
        len(bad_paths),
    )
