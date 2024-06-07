import requests
import multiprocessing
from multiprocessing import Pool
import pandas as pd
import os


def download_image(img_name, url):
    save_folder = r"C:\Code\ML\Image\Card_test\psa_data"
    try:
        response = requests.get(url)
        response.raise_for_status()

        save_path = os.path.join(save_folder, img_name)
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {img_name}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")


if __name__ == "__main__":

    data = pd.read_csv(r"C:\Code\ML\Text\psa1.csv")
    img_info = []
    for i in range(13840, 14000):
        img_name = data[data.columns[0]][i] + '.jpg'
        if (img_name.count('/') > 0 or img_name.count('\\') > 0
                or img_name.count('?') > 0) or img_name.count('"') > 0:
            continue
        img_url = "http://100.64.0.31:9000/" + data[data.columns[1]][i]
        img_info.append((img_name, img_url))
    # video_links = [
    #     "http://example.com/image1.jpg",
    #     "http://example.com/image2.png",
    #     # 添加更多图片链接
    # ]

    # 创建一个进程池
    pool = Pool(processes=4)

    # 使用进程池并行执行下载任务
    pool.starmap(download_image, img_info)

    # 关闭进程池
    pool.close()
    pool.join()
