import os
import cv2
import math
import time
from tqdm import tqdm
from MyBatchOnnxYolo import MyBatchOnnxYolo

# ------------------------------------------------------
# 配置文件
# ------------------------------------------------------
INPUT_DIR = r"/home/martin/ML/Image/CardCls/pokemon_cn"
OUTPUT_DIR = r"/home/martin/ML/Image/CardCls/pokemon_cn_224/train"
MODEL_PATH = r"/home/martin/ML/Model/card_cls/yolov11n_card_seg01.onnx"

TARGET_SIZE = (224, 224)   # (width, height)
BATCH_SIZE = 32
YOLO_IMG_SIZE = 640

# 支持的图片后缀
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def save_image_as_jpg(output_path_without_ext, img_bgr, jpg_quality=95):
    """
    统一保存为 jpg，兼容中文路径
    output_path_without_ext: 不带扩展名，或带任意扩展名都行
    """
    base, _ = os.path.splitext(output_path_without_ext)
    output_jpg_path = base + ".jpg"

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality]
    success, encoded_img = cv2.imencode(".jpg", img_bgr, encode_param)
    if not success:
        raise ValueError(f"cv2.imencode 编码失败: {output_jpg_path}")

    encoded_img.tofile(output_jpg_path)
    return output_jpg_path

def scan_images_recursive(input_dir):
    """
    递归扫描所有图片，返回:
    [
        {
            "full_path": 完整路径,
            "rel_path": 相对 INPUT_DIR 的相对路径,
            "filename": 文件名
        },
        ...
    ]
    """
    image_items = []

    for root, _, files in os.walk(input_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in VALID_EXTS:
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, input_dir)
                image_items.append({
                    "full_path": full_path,
                    "rel_path": rel_path,
                    "filename": fname
                })

    return image_items


def process_images():
    print("--- 开始处理图片 ---")
    print(f"输入文件夹: {INPUT_DIR}")
    print(f"输出文件夹: {OUTPUT_DIR}")
    print(f"模型路径: {MODEL_PATH}")
    print(f"目标尺寸: {TARGET_SIZE}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"YOLO推理尺寸: {YOLO_IMG_SIZE}")

    # --- 1. 创建输出文件夹 ---
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"输出文件夹 '{OUTPUT_DIR}' 已检查/创建。")
    except OSError as e:
        print(f"错误：无法创建输出文件夹 '{OUTPUT_DIR}'，错误信息: {e}")
        return

    # --- 2. 递归扫描输入文件夹 ---
    try:
        image_items = scan_images_recursive(INPUT_DIR)
        total_files = len(image_items)

        if total_files == 0:
            print(f"警告：在 '{INPUT_DIR}' 中没有找到支持的图片文件。")
            print(f"支持格式: {sorted(list(VALID_EXTS))}")
            return

        print(f"找到 {total_files} 个图片文件。")
    except FileNotFoundError:
        print(f"错误：输入文件夹 '{INPUT_DIR}' 不存在。")
        return
    except Exception as e:
        print(f"错误：扫描输入文件夹时出错。错误信息: {e}")
        return

    # --- 3. 初始化模型 ---
    try:
        print("正在加载 YOLO 模型...")
        model = MyBatchOnnxYolo(MODEL_PATH, task='segment')
        print("YOLO 模型加载成功。")
    except FileNotFoundError:
        print(f"错误: 模型文件未找到于 '{MODEL_PATH}'")
        return
    except Exception as e:
        print(f"错误：加载 YOLO 模型失败。错误信息: {e}")
        return

    # --- 4. 分批处理 ---
    num_batches = math.ceil(total_files / BATCH_SIZE)
    print(f"将分为 {num_batches} 个批次进行处理。")

    start_time = time.time()
    processed_count = 0
    error_count = 0

    with tqdm(total=total_files, desc="处理图片", unit="img") as pbar:
        for i in range(0, total_files, BATCH_SIZE):
            batch_start_time = time.time()

            current_batch_items = image_items[i:min(i + BATCH_SIZE, total_files)]
            current_batch_paths = [item["full_path"] for item in current_batch_items]

            if not current_batch_paths:
                continue

            try:
                # --- 5. YOLO 批处理推理 ---
                model.predict_batch(current_batch_paths, imgsz=YOLO_IMG_SIZE)

                # --- 6. 获取裁剪后的图像列表 ---
                processed_batch_images = model.get_max_img_list()

                # 安全检查，防止顺序或数量不一致
                if len(processed_batch_images) != len(current_batch_items):
                    raise ValueError(
                        f"YOLO输出数量与输入数量不一致: "
                        f"input={len(current_batch_items)}, output={len(processed_batch_images)}"
                    )

                # --- 7. 缩放和保存 ---
                for idx_in_batch, img_rgb in enumerate(processed_batch_images):
                    item = current_batch_items[idx_in_batch]
                    rel_path = item["rel_path"]
                    full_path = item["full_path"]

                    if img_rgb is None:
                        print(f"警告：无法处理图片 {full_path}，已跳过。")
                        error_count += 1
                        pbar.update(1)
                        continue

                    try:
                        # 缩放
                        resized_img_rgb = cv2.resize(img_rgb, TARGET_SIZE)

                        # RGB -> BGR
                        resized_img_bgr = cv2.cvtColor(resized_img_rgb, cv2.COLOR_RGB2BGR)

                        # 构造输出路径，并保持原始目录结构
                        output_path = os.path.join(OUTPUT_DIR, rel_path)
                        output_dir = os.path.dirname(output_path)
                        os.makedirs(output_dir, exist_ok=True)

                        # 保存
                        saved_path = save_image_as_jpg(output_path, resized_img_bgr)
                        processed_count += 1

                    except Exception as e_inner:
                        print(f"错误：处理或保存图片 {full_path} 时出错: {e_inner}")
                        error_count += 1
                    finally:
                        pbar.update(1)

            except Exception as e_batch:
                print(f"\n错误：处理批次 {i // BATCH_SIZE + 1}/{num_batches} 时发生严重错误: {e_batch}")
                if current_batch_items:
                    print(f"  涉及文件范围: {current_batch_items[0]['full_path']} ... {current_batch_items[-1]['full_path']}")
                batch_error_num = len(current_batch_items)
                error_count += batch_error_num
                pbar.update(batch_error_num)

            batch_end_time = time.time()
            print(f"批次 {i // BATCH_SIZE + 1}/{num_batches} 处理完成，耗时: {batch_end_time - batch_start_time:.2f} 秒")

    end_time = time.time()
    total_time = end_time - start_time

    print("\n--- 处理完成 ---")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"成功处理并保存文件数: {processed_count}")
    print(f"处理失败或跳过文件数: {error_count}")
    if total_files > 0 and total_time > 0:
        print(f"平均处理速度: {total_files / total_time:.2f} 张图片/秒")


if __name__ == "__main__":
    process_images()