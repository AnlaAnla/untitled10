import pandas as pd
import numpy as np
import requests
import os
from multiprocessing import Pool

data = pd.read_csv(r"C:\Code\ML\Text\pokemon_csv\train.csv")

# 创建文件夹来保存下载的图片
save_folder = r"C:\Code\ML\Image\card_cls\All_Pokemon_Card\data01"
# os.makedirs(save_folder, exist_ok=True)


def download_image(args):
    """
    下载图片并保存到文件
    """
    image_url, img_name = args
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # 检查响应状态码
        with open(os.path.join(save_folder, img_name), 'wb') as f:
            f.write(response.content)
        print(f'Downloaded: {img_name}')
    except requests.exceptions.RequestException as e:
        print(f'Failed to download: {image_url}: {e}')


def download_images():
    """
    使用多进程下载图片
    """
    args = []
    for i in range(data.shape[0]):
        id_num = '#' + str(i + 1)
        img_id = data['id'][i]
        set_name = data['set_name'][i]
        name = data['name'][i]
        image_url = data['image_url'][i]
        img_name = f"{id_num}_{img_id}_{set_name}_{name}.jpg"
        args.append((image_url, img_name))

    # 创建进程池并并行下载
    with Pool() as pool:
        pool.map(download_image, args)


if __name__ == '__main__':
    download_images()
