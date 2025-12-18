import os

# 【关键】指定显卡 ID 为 2
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from transformers import ViTModel, ViTConfig
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import List, Union
import logging
import time
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ==========================================
# 你的原始类定义 (保持不变)
# ==========================================
class MyViTFeatureExtractor:
    def __init__(self, local_model_path: str) -> None:
        if not os.path.isdir(local_model_path):
            raise NotADirectoryError(f"Model path not found: {local_model_path}")

        logging.info(f"Loading model from: {local_model_path}")

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        logging.info(f"Using device: {self.device}")

        try:
            self.config = ViTConfig.from_pretrained(local_model_path, local_files_only=True)
            self.model = ViTModel.from_pretrained(local_model_path, config=self.config, local_files_only=True)
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.feature_dim = self.model.config.hidden_size

    def run(self, imgs: List[Union[str, np.ndarray, Image.Image]], normalize: bool = True) -> np.ndarray:
        # (原有的 run 方法保留，但在本次压力测试中我们直接调用 model 以跳过 CPU IO 瓶颈)
        pass

    # ==========================================


# 压力测试逻辑
# ==========================================
def benchmark_throughput(model_path):
    print(f"\n{'=' * 60}")
    print(f"🚀 开始 ViT-Base 显存与速度基准测试")
    print(f"📍 目标显卡: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"{'=' * 60}\n")

    # 1. 初始化模型
    try:
        extractor = MyViTFeatureExtractor(model_path)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    model = extractor.model
    device = extractor.device

    # 2. 定义测试范围: 1, 100, 200, ..., 1200
    batch_sizes = [1] + list(range(10, 201, 20))

    print(f"{'Batch Size':<12} | {'耗时 (ms)':<12} | {'吞吐量 (img/s)':<18} | {'显存占用 (GB)':<15} | {'状态':<10}")
    print("-" * 80)

    for batch_size in batch_sizes:
        try:
            # --- A. 准备数据 (模拟 Tensor，不计入推理时间) ---
            # 形状: [Batch, 3, 224, 224]
            # 我们使用半精度(FP16)还是单精度(FP32)取决于你的实际场景，这里默认用 FP32 (Torch默认)
            # 如果想测试 FP16，可以将 inputs 转为 .half()
            inputs = torch.randn(batch_size, 3, 224, 224, device=device)

            # 清理之前的缓存，确保显存读数准确
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # --- B. 预热 (Warm up) ---
            # GPU 首次运行会有 overhead，先跑一次不计时的
            with torch.no_grad():
                _ = model(inputs)

            torch.cuda.synchronize()  # 等待预热完成

            # --- C. 计时推理 ---
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            with torch.no_grad():
                # 模拟你的 run 方法中的核心逻辑
                outputs = model(inputs)
                features = outputs.last_hidden_state[:, 0, :]
                # 如果做了 normalize，通常开销很小，这里主要测模型 forward
                features = F.normalize(features, p=2, dim=1)
            end_event.record()

            torch.cuda.synchronize()  # 等待 GPU 完成所有计算

            # --- D. 计算统计 ---
            elapsed_time_ms = start_event.elapsed_time(end_event)  # 毫秒
            images_per_sec = batch_size / (elapsed_time_ms / 1000.0)

            # 获取显存峰值
            max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB

            print(
                f"{batch_size:<12} | {elapsed_time_ms:<12.2f} | {images_per_sec:<18.2f} | {max_memory:<15.2f} | ✅ 成功")

            # 主动释放显存
            del inputs, outputs, features

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"{batch_size:<12} | {'-':<12} | {'-':<18} | {'FAIL':<15} | ❌ 显存溢出 (OOM)")
                torch.cuda.empty_cache()  # 尝试恢复
                break  # 已经炸显存了，后面更大的肯定也不行，直接停止
            else:
                print(f"❌ 未知错误 (Batch {batch_size}): {e}")
                break

    print("-" * 80)
    print("\n💡 提示:")
    print("1. '吞吐量' 越高，说明显卡利用率越好。")
    print("2. 实际工程中，建议选择最大可用 Batch Size 的 70%-80%，防止显存碎片化或 YOLO 进程抢占。")


if __name__ == "__main__":
    # 请修改为你的实际模型路径
    MODEL_PATH = "/home/martin/ML/Model/pokemon_cls/vit-base-patch16-224-Pokemon03"

    benchmark_throughput(MODEL_PATH)