import os
import threading
from PIL import Image
from glob import glob
from tqdm import tqdm


def resize_image(image_path, output_shape, output_dir):
    """
    调整图片大小并保存到指定目录。

    Args:
        image_path: 图片的完整路径。
        output_shape: 目标尺寸 (width, height)。
        output_dir: 保存调整大小后的图片的目录。
    """
    try:
        img = Image.open(image_path)
        if img.size != output_shape:
            img = img.resize(output_shape)

        # 构建输出路径
        relative_path = os.path.relpath(image_path, start=input_dir)
        output_path = os.path.join(output_dir, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 创建必要的目录

        img.save(output_path)
        # print(f"已压缩: {image_path} -> {output_path}")
    except Exception as e:
        print(f"处理图片失败: {image_path}, 错误: {e}")


def process_images_multithreaded(image_paths, output_shape, output_dir, num_threads=4):
    """
    使用多线程处理图片。

    Args:
        image_paths: 图片路径列表。
        output_shape: 目标尺寸 (width, height)。
        output_dir: 保存调整大小后的图片的目录。
        num_threads: 线程数。
    """
    with tqdm(total=len(image_paths), desc="处理进度") as pbar:
        threads = []
        for image_path in image_paths:
            thread = threading.Thread(target=resize_image, args=(image_path, output_shape, output_dir))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
            pbar.update(1)


# 指定输入目录和 glob 模式
input_dir = r"D:\Code\ML\Image\_CLASSIFY\card_cls2\Serices_cls_data_yolo224\train"
glob_pattern = os.path.join(input_dir, "*", "*")  # 可根据需要修改文件扩展名
print(glob_pattern)

# 指定输出目录
output_dir = r"D:\Code\ML\Image\_CLASSIFY\card_cls2\Serices_cls_data_yolo224\train_resized"

# 指定目标尺寸
target_shape = (224, 224)

# 获取所有匹配的图片路径
image_paths = glob(glob_pattern, recursive=True)

# 使用多线程处理图片
process_images_multithreaded(image_paths, target_shape, output_dir)

print("图片处理完成！")
