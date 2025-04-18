import os
import cv2
import numpy as np
from tqdm import tqdm
import math
import time
from Tool.MyBatchOnnxYolo import MyBatchOnnxYolo

# ------------------------------------------------------
# 配置文件
# ------------------------------------------------------
# --- 输入和输出路径 ---
INPUT_DIR = r"D:\Code\ML\Embedding\img_vec\image"
OUTPUT_DIR = r"D:\Code\ML\Embedding\img_vec\image_yolo224"
MODEL_PATH = r"D:\Code\ML\Model\Card_Seg\yolov11n_card_seg01.pt"  # 使用 .pt 模型

# --- 模型和处理参数 ---
TARGET_SIZE = (224, 224)  # 目标缩放尺寸 (宽度, 高度)
BATCH_SIZE = 32  # 每个批次处理的图像数量 (为 3070 建议 16, 32 或 64，先试试 32)
YOLO_IMG_SIZE = 640  # YOLO 模型推理时使用的图像尺寸


# ------------------------------------------------------

def process_images():
    """
    主处理函数：加载模型，遍历图片，进行批处理、裁剪、缩放和保存。
    """
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
        print(f"错误：无法创建输出文件夹 '{OUTPUT_DIR}'. 错误信息: {e}")
        return

    # --- 2. 扫描输入文件夹 ---
    try:
        all_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.jpg')]
        image_paths = [os.path.join(INPUT_DIR, f) for f in all_files]
        total_files = len(image_paths)
        if total_files == 0:
            print(f"警告：在 '{INPUT_DIR}' 中没有找到 .jpg 文件。")
            return
        print(f"找到 {total_files} 个 .jpg 文件。")
    except FileNotFoundError:
        print(f"错误：输入文件夹 '{INPUT_DIR}' 不存在。")
        return
    except Exception as e:
        print(f"错误：扫描输入文件夹时出错。错误信息: {e}")
        return

    # --- 3. 初始化模型 ---
    try:
        print("正在加载 YOLO 模型...")
        model = MyBatchOnnxYolo(MODEL_PATH, task='segment')  # task='segment' 或 'detect' 取决于你的模型
        print("YOLO 模型加载成功。")
    except FileNotFoundError:
        print(f"错误: 模型文件未找到于 '{MODEL_PATH}'")
        return
    except Exception as e:
        print(f"错误：加载 YOLO 模型失败。错误信息: {e}")
        # 可能需要安装特定依赖: pip install ultralytics onnxruntime-gpu (如果使用ONNX)
        return

    # --- 4. 分批处理 ---
    num_batches = math.ceil(total_files / BATCH_SIZE)
    print(f"将分为 {num_batches} 个批次进行处理。")

    start_time = time.time()
    processed_count = 0
    error_count = 0

    # 使用 tqdm 创建进度条
    with tqdm(total=total_files, desc="处理图片", unit="img") as pbar:
        for i in range(0, total_files, BATCH_SIZE):
            batch_start_time = time.time()

            # 获取当前批次的路径和文件名
            current_batch_paths = image_paths[i:min(i + BATCH_SIZE, total_files)]
            current_batch_filenames = all_files[i:min(i + BATCH_SIZE, total_files)]

            if not current_batch_paths:
                continue

            try:
                # --- 5. YOLO 批处理推理 ---
                model.predict_batch(current_batch_paths, imgsz=YOLO_IMG_SIZE)

                # --- 6. 提取最大卡牌列表 ---
                # get_max_img_list 会返回裁剪后的卡牌 (RGB) 或原始图像 (RGB)
                processed_batch_images = model.get_max_img_list()

                # --- 7. 缩放和保存 ---
                for idx_in_batch, img_rgb in enumerate(processed_batch_images):
                    if img_rgb is None:
                        # 处理 get_max_img_list 返回 None 的情况 (例如原图无效)
                        print(
                            f"警告：无法处理图片 {current_batch_filenames[idx_in_batch]} (可能原始文件无效或损坏)，已跳过。")
                        error_count += 1
                        pbar.update(1)  # 更新进度条计数
                        continue

                    try:
                        # 缩放图像
                        # cv2.resize 需要 (宽度, 高度)
                        resized_img_rgb = cv2.resize(img_rgb, TARGET_SIZE)

                        # 构建输出路径
                        original_filename = current_batch_filenames[idx_in_batch]
                        output_path = os.path.join(OUTPUT_DIR, original_filename)

                        # 转换回 BGR 以便 cv2.imwrite 保存
                        resized_img_bgr = cv2.cvtColor(resized_img_rgb, cv2.COLOR_RGB2BGR)

                        # 保存图像
                        cv2.imwrite(output_path, resized_img_bgr)
                        processed_count += 1

                    except Exception as e_inner:
                        print(f"错误：处理或保存图片 {current_batch_filenames[idx_in_batch]} 时出错: {e_inner}")
                        error_count += 1
                    finally:
                        pbar.update(1)  # 确保无论成功失败都更新进度条

            except Exception as e_batch:
                print(f"\n错误：处理批次 {i // BATCH_SIZE + 1}/{num_batches} 时发生严重错误: {e_batch}")
                print(f"  涉及的文件范围: {current_batch_filenames[0]} ... {current_batch_filenames[-1]}")
                # 将此批次中的所有文件计为错误
                batch_error_num = len(current_batch_paths)
                error_count += batch_error_num
                pbar.update(batch_error_num)  # 更新进度条以跳过此批次

            batch_end_time = time.time()
            # 可选：打印每个批次的处理时间
            print(f"批次 {i//BATCH_SIZE + 1}/{num_batches} 处理完成，耗时: {batch_end_time - batch_start_time:.2f} 秒")

    end_time = time.time()
    total_time = end_time - start_time

    print("\n--- 处理完成 ---")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"成功处理并保存文件数: {processed_count}")
    print(f"处理失败或跳过文件数: {error_count}")
    if total_files > 0 and total_time > 0:
        print(f"平均处理速度: {total_files / total_time:.2f} 张图片/秒")


# --- 脚本入口 ---
if __name__ == "__main__":
    process_images()
