import pandas as pd
import requests
import os
import time


def download_image(img_url, save_dir, timeout=10, max_retries=3):
    """
    下载单个图片并保存，跳过已存在的文件。

    Args:
        img_url (str): 图片的 URL。
        save_dir (str): 保存图片的目录。
        timeout (int): 请求超时时间（秒）。
        max_retries (int): 最大重试次数

    Returns:
        bool: 下载成功返回 True，失败返回 False。
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
        print(f"Skipping already downloaded file: {file_name}")
        return True  # 文件已存在，视为下载成功

    for attempt in range(max_retries):
        try:
            response = requests.get(img_url, timeout=timeout)
            response.raise_for_status()

            os.makedirs(save_dir, exist_ok=True)  # 确保目录存在

            with open(file_path, 'wb') as f:
                f.write(response.content)
            return True

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))  # 递增等待时间
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return False

    print(f"Failed to download {img_url} after {max_retries} attempts.")
    return False


def main(data_path, image_dir):
    """
    主函数：读取 Excel 文件，下载图片。

    Args:
        data_path (str): Excel 文件的路径。
        image_dir (str): 保存图片的目录。
    """
    try:
        df = pd.read_excel(data_path)
        img_urls = df['img_origin'].dropna().unique().tolist()  # 获取不重复的图片 URL
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    os.makedirs(image_dir, exist_ok=True)  # 确保图片保存目录存在

    downloaded_count = 0
    total_images = len(img_urls)

    for i, img_url in enumerate(img_urls):
        print(f"Downloading image {i + 1}/{total_images}: {img_url}")
        if download_image(img_url, image_dir):
            downloaded_count += 1

    print(f"Download complete. {downloaded_count}/{total_images} images downloaded.")


if __name__ == "__main__":
    DATA_PATH = "D:\\Code\\ML\\Embedding\\img_vec\\checklist_ebay_data2023.xlsx"  # 替换为你的 Excel 文件路径
    IMAGE_DIR = "D:\\Code\\ML\\Embedding\\img_vec\\image"  # 替换为你希望保存图片的目录
    main(DATA_PATH, IMAGE_DIR)