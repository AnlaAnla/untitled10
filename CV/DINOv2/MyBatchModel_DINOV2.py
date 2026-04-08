import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model
from safetensors.torch import load_file
import numpy as np
import os
import cv2
import logging
from typing import List, Union
from PIL import Image

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MyDinoHFExtractor:
    def __init__(self, model_dir: str):
        """
        Args:
            model_dir: 包含 config.json 和 model.safetensors 的文件夹路径
        """
        if not os.path.isdir(model_dir):
            raise NotADirectoryError(f"Directory not found: {model_dir}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Initializing HF DINOv2 from {model_dir} on {self.device}...")

        # 1. 加载 Hugging Face DINOv2 主干
        try:
            self.backbone = Dinov2Model.from_pretrained(model_dir, local_files_only=True)
        except Exception as e:
            logging.error(f"Failed to load HF model: {e}")
            raise

        # 2. 恢复 BN Neck 层
        self.feature_dim = self.backbone.config.hidden_size
        self.bn_neck = nn.BatchNorm1d(self.feature_dim)

        # 尝试加载 BN 权重
        st_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(st_path):
            self._load_bn_weights(st_path)
        else:
            logging.warning(f"Weights file not found: {st_path}")

        # 3. 设置模型状态
        self.backbone.to(self.device).eval()
        self.bn_neck.to(self.device).eval()

        # 4. 获取配置的图像尺寸 (392)
        self.image_size = self.backbone.config.image_size
        logging.info(f"Model Image Size: {self.image_size}")

        # 预定义 Mean 和 Std (ImageNet 格式)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _load_bn_weights(self, safetensors_path):
        """从 safetensors 文件中手动加载 bn_neck 权重"""
        try:
            full_state = load_file(safetensors_path, device="cpu")
            bn_dict = {}
            for k, v in full_state.items():
                if k.startswith("bn_neck."):
                    bn_dict[k.replace("bn_neck.", "")] = v

            if bn_dict:
                self.bn_neck.load_state_dict(bn_dict)
                logging.info("✅ BN Neck weights loaded successfully.")
            else:
                logging.warning("⚠️ CRITICAL: No BN Neck weights found in safetensors!")
        except Exception as e:
            logging.warning(f"Error loading BN weights: {e}")

    def _preprocess_letterbox(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        【关键】复刻 yolo_batch_process.py 中的 Letterbox 逻辑
        Input: RGB Numpy Array (H, W, 3)
        Output: Normalized Tensor numpy (3, H, W)
        """
        shape = img_rgb.shape[:2]  # [h, w]
        target_size = self.image_size
        new_shape = (target_size, target_size)

        # 1. 计算缩放比例 (保持长宽比)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # 计算 Resize 后的尺寸 (unpad)
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

        # 计算 Padding 大小
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2  # 分成两半用于左右/上下填充
        dh /= 2

        # 2. Resize (步骤1)
        if shape[::-1] != new_unpad:
            # 严格使用 INTER_CUBIC，和数据预处理脚本一致
            img_rgb = cv2.resize(img_rgb, new_unpad, interpolation=cv2.INTER_CUBIC)

        # 3. Add Border (Padding)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        # 填充黑色边框 (0,0,0)
        img_rgb = cv2.copyMakeBorder(img_rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # 4. Resize
        # 这一步在 yolo_batch_process.py 里也有，确保最终尺寸绝对精确
        if img_rgb.shape[:2] != new_shape:
            img_rgb = cv2.resize(img_rgb, new_shape, interpolation=cv2.INTER_CUBIC)

        # 5. Normalize (ImageNet)
        # 归一化: (Pixel/255 - Mean) / Std
        img_float = img_rgb.astype(np.float32) / 255.0
        img_norm = (img_float - self.mean) / self.std

        # 6. HWC -> CHW
        img_chw = img_norm.transpose(2, 0, 1)
        return img_chw

    @torch.no_grad()
    def run(self, imgs: List[Union[str, np.ndarray, Image.Image]], normalize: bool = True) -> np.ndarray:
        """
        处理一批图像并返回特征向量。
        Args:
            imgs: 支持 图像路径str, Numpy数组(RGB), PIL Image
            normalize: 是否进行 L2 归一化 (默认 True)
        Returns:
            Numpy array (Batch_Size, Feature_Dim)
        """
        if not isinstance(imgs, list):
            logging.error("Input must be a list of images.")
            raise TypeError("Input must be a list of images.")

        if not imgs:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        tensors = []
        valid_indices = []

        for i, img_input in enumerate(imgs):
            try:
                img_rgb = None

                # --- 1. 统一读取为 RGB Numpy ---
                if isinstance(img_input, str):
                    if not os.path.exists(img_input):
                        logging.error(f"File not found: {img_input}")
                        continue
                    # 读取为 BGR 转 RGB (保持和 cv2.imread 一致的解码)
                    img_bgr = cv2.imread(img_input)
                    if img_bgr is None:
                        logging.error(f"Failed to read image: {img_input}")
                        continue
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                elif isinstance(img_input, np.ndarray):
                    # 假设输入是 RGB
                    if img_input.ndim != 3 or img_input.shape[2] != 3:
                        logging.warning(f"Invalid numpy shape {img_input.shape}, skipping.")
                        continue

                    # 确保是 uint8
                    if img_input.dtype != np.uint8:
                        if img_input.max() <= 1.0:
                            img_input = (img_input * 255).astype(np.uint8)
                        else:
                            img_input = img_input.astype(np.uint8)
                    img_rgb = img_input

                elif isinstance(img_input, Image.Image):
                    # PIL -> RGB Numpy
                    img_rgb = np.array(img_input.convert('RGB'))

                else:
                    logging.warning(f"Unsupported input type: {type(img_input)}")
                    continue

                # --- 2. Letterbox 预处理 ---
                if img_rgb is not None:
                    processed = self._preprocess_letterbox(img_rgb)
                    tensors.append(torch.from_numpy(processed))
                    valid_indices.append(i)

            except Exception as e:
                logging.error(f"Error processing image index {i}: {e}")

        if not tensors:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        # --- 3. 模型推理 ---
        batch_tensor = torch.stack(tensors).float().to(self.device)

        # Backbone
        outputs = self.backbone(batch_tensor)
        cls_token = outputs.last_hidden_state[:, 0, :]

        # BN Neck
        feat = self.bn_neck(cls_token)

        # Normalize (L2)
        if normalize:
            feat = F.normalize(feat, p=2, dim=1)

        feat_np = feat.cpu().numpy()

        # --- 4. 保持输出长度与输入一致 (填充 NaN) ---
        if len(feat_np) != len(imgs):
            final_output = np.full((len(imgs), self.feature_dim), np.nan, dtype=np.float32)
            for idx, vec in zip(valid_indices, feat_np):
                final_output[idx] = vec
            return final_output
        else:
            return feat_np
