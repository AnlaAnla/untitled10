import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
from multiprocessing import cpu_count

# --- 假设你的类保存在 yolo_utils.py 中，如果不是，请将你的类代码直接贴在这里 ---
# from yolo_utils import MyBatchOnnxYolo
# 为了方便你直接运行，我这里直接包含你的类定义(稍微精简引用以适配脚本)
from ultralytics import YOLO
from typing import List, Union, Optional
import PIL.Image


# ==========================================
# 1. 你的 YOLO 类定义 (保持不变)
# ==========================================
class MyBatchOnnxYolo:
    def __init__(self, model_path: str, task: str = 'segment', verbose: bool = False):
        self.model = YOLO(model_path, task=task, verbose=verbose)
        self.results = None
        self.batch_size = 0

    def predict_batch(self, image_list, imgsz=640, **kwargs):
        if not image_list:
            self.results = []
            self.batch_size = 0
            return
        self.results = self.model.predict(image_list, verbose=False, imgsz=imgsz, **kwargs)
        self.batch_size = len(self.results)

    def get_max_img_list(self, cls_id: int = 0) -> List[Optional[np.ndarray]]:
        if self.results is None:
            raise ValueError("Must call predict_batch() first.")

        processed_images = []
        for i in range(self.batch_size):
            res = self.results[i]
            orig_img = res.orig_img
            boxes = res.boxes

            # 默认使用原图 (注意：YOLO内部 orig_img 是 BGR)
            # 你的原始代码这里转成了 RGB，我们在后续处理时要注意转回来，或者直接在这里处理
            # 为了兼容你的类逻辑，我们保持你的输出为 RGB
            final_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB) if orig_img is not None else None

            if boxes is not None and len(boxes) > 0 and boxes.cls is not None and cls_id in boxes.cls.cpu():
                max_area = 0.0
                max_box = None
                xyxy = boxes.xyxy.cpu().numpy()
                clss = boxes.cls.cpu().numpy()

                for k, box in enumerate(xyxy):
                    if clss[k] != cls_id: continue
                    x1, y1, x2, y2 = box
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        max_box = box

                if max_box is not None:
                    x1, y1, x2, y2 = map(int, max_box)
                    h, w = orig_img.shape[:2]
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                    if x2 > x1 and y2 > y1:
                        crop = orig_img[y1:y2, x1:x2]
                        final_img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            processed_images.append(final_img)
        return processed_images


# ==========================================
# 2. 辅助函数: Letterbox Resize (静态函数以便多进程调用)
# ==========================================
def letterbox_resize_and_save(args):
    """
    接收 (RGB图像数组, 保存路径, 目标尺寸)
    进行 Padding Resize 并保存为文件
    """
    img_rgb, save_path, target_size = args

    if img_rgb is None:
        return False

    try:
        # !!! 关键点 !!!
        # 你的 YOLO 类返回的是 RGB，但 OpenCV imwrite 需要 BGR
        # 所以这里必须转回 BGR，否则图片颜色会变蓝
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # --- 开始 Letterbox 逻辑 ---
        shape = img_bgr.shape[:2]  # [h, w]
        new_shape = (target_size, target_size)

        # 计算缩放比例
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

        # 计算 Padding
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        # Resize
        if shape[::-1] != new_unpad:
            img_bgr = cv2.resize(img_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)

        # Padding (黑色)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_final = cv2.copyMakeBorder(img_bgr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # 最终强制 Resize (防止1像素误差)
        img_final = cv2.resize(img_final, new_shape, interpolation=cv2.INTER_LINEAR)

        # 确保存储目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存
        cv2.imwrite(save_path, img_final)
        return True
    except Exception as e:
        print(f"Error saving {save_path}: {e}")
        return False


# ==========================================
# 3. 主流程脚本
# ==========================================
def main():
    # --- 配置区域 ---
    SRC_ROOT = Path("/home/martin/ML/Image/CardCls/panini_archive")
    DST_ROOT = Path("/home/martin/ML/Image/CardCls/panini_archive_resize384/train")
    MODEL_PATH = r"/home/martin/ML/Model/card_cls/yolov11n_card_seg01.onnx"

    TARGET_SIZE = 384
    BATCH_SIZE = 32  # 显存够大可开 64 或 128
    NUM_WORKERS = 8  # CPU 处理保存任务的进程数

    # 1. 初始化模型
    print("正在加载 YOLO 模型...")
    # 注意：task='segment' 还是 'detect' 取决于你的模型，你之前代码写的是 segment
    yolo_wrapper = MyBatchOnnxYolo(MODEL_PATH, task='segment')

    # 2. 扫描文件
    print(f"正在扫描目录: {SRC_ROOT}")
    tasks = []

    # 遍历: SRC_ROOT -> 中文件夹 -> 标签文件夹 -> 图片
    for mid_folder in SRC_ROOT.iterdir():
        if not mid_folder.is_dir(): continue

        for label_folder in mid_folder.iterdir():
            if not label_folder.is_dir(): continue

            # 拿到 Label 名称 (如 "2023, Black, Base Amethyst...")
            label_name = label_folder.name

            # 假设该文件夹下有图片
            files = list(label_folder.iterdir())
            if not files: continue  # 空文件夹跳过

            for img_file in files:
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
                    # 构造目标路径: .../train/Label_Name/Image.jpg
                    # 这里成功去掉了中间文件夹
                    target_path = DST_ROOT / label_name / img_file.name

                    tasks.append({
                        'src': str(img_file),
                        'dst': str(target_path)
                    })

    total_files = len(tasks)
    print(f"扫描完成，共发现 {total_files} 张图片。准备开始批处理...")

    # 3. 批处理循环
    # 使用 ProcessPoolExecutor 进行后台的 Resize 和 Save，不阻塞 GPU 推理
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:

        # 按 Batch 分块
        for i in tqdm(range(0, total_files, BATCH_SIZE), desc="Batch Processing"):
            batch_tasks = tasks[i: i + BATCH_SIZE]

            # --- Step A: 读取图片 (多线程加速 IO) ---
            # 使用简单的列表推导式或线程池读取
            batch_imgs = []
            valid_tasks = []  # 记录读取成功的任务

            for task in batch_tasks:
                try:
                    # imread 读取的是 BGR
                    img = cv2.imread(task['src'])
                    if img is None or img.size == 0:
                        print(f"[坏图警告] 无法读取: {task['src']}")
                        continue
                    batch_imgs.append(img)
                    valid_tasks.append(task)
                except Exception:
                    continue

            if not batch_imgs: continue

            # --- Step B: YOLO Batch 推理 (GPU) ---
            # 这一步调用你的类
            yolo_wrapper.predict_batch(batch_imgs, imgsz=640, conf=0.25)

            # 获取裁剪后的图片列表 (注意：你的类返回的是 RGB numpy list)
            cropped_imgs_rgb = yolo_wrapper.get_max_img_list(cls_id=0)

            # --- Step C: 提交保存任务 (CPU 多进程) ---
            # 准备参数列表
            save_jobs = []
            for j, crop_rgb in enumerate(cropped_imgs_rgb):
                dst_path = valid_tasks[j]['dst']
                save_jobs.append((crop_rgb, dst_path, TARGET_SIZE))

            # 异步提交，不等待保存完成即可开始下一轮 GPU 推理
            # list() 强制提交任务
            list(executor.map(letterbox_resize_and_save, save_jobs))

    print("\n========================================")
    print(f"处理完成！所有数据已保存至: {DST_ROOT}")
    print("========================================")


if __name__ == "__main__":
    # 防止 OpenCV 多线程与 Python 多进程冲突
    try:
        cv2.setNumThreads(0)
    except:
        pass

    # Windows 下必须有 freeze_support，Linux 其实不需要但加上无妨
    from multiprocessing import freeze_support

    freeze_support()

    main()