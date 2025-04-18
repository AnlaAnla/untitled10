import pandas as pd
import requests
import os
import time
import concurrent.futures
from tqdm import tqdm  # 导入 tqdm 库


def download_image(img_url, save_dir, timeout=10, max_retries=3):
    """
    下载单个图片并保存，跳过已存在的文件。

    Args:
        img_url (str): 图片的 URL。
        save_dir (str): 保存图片的目录。
        timeout (int): 请求超时时间（秒）。
        max_retries (int): 最大重试次数。

    Returns:
        bool: 下载成功返回 True，失败返回 False。  (如果跳过已存在文件，也返回 True)
    """
    if not img_url.startswith("http"):
        img_url = "https:" + img_url

    try:
        file_name_base = img_url.split("/")[-2]
        file_name = file_name_base + ".jpg"
    except IndexError:
        print(f"Error: Could not extract filename from URL: {img_url}")
        return False

    file_path = os.path.join(save_dir, file_name)

    # 检查文件是否已存在
    if os.path.exists(file_path):
        # print(f"Skipping already downloaded file: {file_name}") # 不再打印，由 tqdm 处理
        return True

    for attempt in range(max_retries):
        try:
            response = requests.get(img_url, timeout=timeout)
            response.raise_for_status()

            os.makedirs(save_dir, exist_ok=True)

            with open(file_path, 'wb') as f:
                f.write(response.content)
            return True

        except requests.exceptions.RequestException as e:
            # print(f"Attempt {attempt + 1}/{max_retries} failed: {e}") # 不再打印，由 tqdm 处理
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
        except Exception as e:
            print(f"An unexpected error occurred: {e}")  # 严重的错误才打印
            return False  # 出现严重错误，不再重试

    # print(f"Failed to download {img_url} after {max_retries} attempts.") # 不再打印，由 tqdm 处理
    return False


def worker(img_url, save_dir):
    """
    线程工作函数：下载单个图片。
    """
    return download_image(img_url, save_dir)


def main(data_path, image_dir, num_threads=10):
    """
    主函数：读取 Excel 文件，使用多线程下载图片。

    Args:
        data_path (str): Excel 文件的路径。
        image_dir (str): 保存图片的目录。
        num_threads (int): 线程数。
    """
    try:
        df = pd.read_excel(data_path)
        img_urls = df['img_origin'].dropna().unique().tolist()
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    os.makedirs(image_dir, exist_ok=True)

    downloaded_count = 0
    total_images = len(img_urls)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 创建任务列表
        futures = [executor.submit(worker, img_url, image_dir) for img_url in img_urls]

        # 使用 tqdm 显示进度条
        with tqdm(total=total_images, desc="Downloading images") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()  # 获取 worker 函数的返回值
                    if result:
                        downloaded_count += 1
                        pbar.update(1)  # 更新进度条
                except Exception as e:
                    print(f"An error occurred during a download: {e}")  # 捕获可能的异常
                    # 这里可以根据需要选择是否继续下载其他图片
                    # 如果不希望因为一个图片的下载错误而停止整个程序，就不要 re-raise 异常

    print(f"\nDownload complete. {downloaded_count}/{total_images} images downloaded.")


if __name__ == "__main__":
    DATA_PATH = "D:\\Code\\ML\\Embedding\\img_vec\\checklist_ebay_data2023.xlsx"
    IMAGE_DIR = "D:\\Code\\ML\\Embedding\\img_vec\\image"
    NUM_THREADS = 10  # 设置线程数
    main(DATA_PATH, IMAGE_DIR, NUM_THREADS)
