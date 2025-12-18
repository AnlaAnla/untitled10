import torch
from transformers import ViTModel, ViTConfig
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import List, Union, Any
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MyViTFeatureExtractor:
    def __init__(self, local_model_path: str) -> None:
        """
        初始化特征提取器。

        Args:
            local_model_path (str): 包含模型配置(config.json)和权重
                                    (model.safetensors 或 pytorch_model.bin)
                                    的本地目录路径。
        """
        if not os.path.isdir(local_model_path):
            logging.error(f"指定的本地模型路径不是一个有效目录: {local_model_path}")
            raise NotADirectoryError(f"指定的本地模型路径不是一个有效目录: {local_model_path}")
        config_path = os.path.join(local_model_path, 'config.json')

        weights_path_st = os.path.join(local_model_path, 'model.safetensors')
        if not os.path.exists(config_path) or not (os.path.exists(weights_path_st)):
            logging.error(
                f"本地模型目录 {local_model_path} 缺少 config.json 或模型权重文件 (model.safetensors / pytorch_model.bin)。")
            raise FileNotFoundError(f"本地模型目录 {local_model_path} 缺少 config.json 或模型权重文件。")

        logging.info(f"从本地目录加载模型: {local_model_path}")

        # 1. 设备选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"使用设备: {self.device}")

        # 2. 从本地目录加载模型配置和权重
        try:
            self.config = ViTConfig.from_pretrained(local_model_path, local_files_only=True)
            self.model = ViTModel.from_pretrained(local_model_path, config=self.config, local_files_only=True)
        except Exception as e:
            logging.error(f"从本地目录 '{local_model_path}' 加载模型失败: {e}")
            raise

        # 3. 模型设置
        self.model.to(self.device)
        self.model.eval()
        logging.info("模型加载完成并设置为评估模式。")

        # 4. 定义图像预处理流程
        self.transform = self._create_inference_transform()

        # 5. 存储特征维度
        self.feature_dim = self.model.config.hidden_size
        logging.info(f"模型特征维度: {self.feature_dim}")

    def _create_inference_transform(self):
        """创建图像预处理的流水线。"""
        image_size = self.config.image_size
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

    def run(self, imgs: List[Union[str, np.ndarray, Image.Image]], normalize: bool = True) -> np.ndarray:
        """
        处理一批图像并返回它们的特征向量 (CLS token 的最后一个隐藏层状态)。
        """
        if not isinstance(imgs, list):
            logging.error("输入必须是一个图像列表。")
            raise TypeError("输入必须是一个图像列表。")

        if not imgs:
            logging.warning("输入图像列表为空。")
            return np.empty((0, self.feature_dim), dtype=np.float32)

        processed_tensors = []
        valid_indices = []

        # 1. 预处理图像
        for i, img_input in enumerate(imgs):
            try:
                if isinstance(img_input, str):
                    if not os.path.exists(img_input):
                        raise FileNotFoundError(f"图像文件未找到: {img_input}")
                    img = Image.open(img_input).convert('RGB')
                elif isinstance(img_input, np.ndarray):
                    if img_input.ndim != 3 or img_input.shape[2] != 3:
                        raise ValueError(f"NumPy 数组必须是 HWC 格式，当前形状: {img_input.shape}")
                    if img_input.dtype != np.uint8:
                        if img_input.max() <= 1.0 and img_input.min() >= 0.0 and img_input.dtype in [np.float32,
                                                                                                     np.float64]:
                            logging.debug("将 float (0-1) NumPy 数组转换为 uint8。")
                            img_input = (img_input * 255).astype(np.uint8)
                        else:
                            logging.warning(
                                f"NumPy 数组 dtype 不是 uint8 ({img_input.dtype})，将尝试强制转换。建议输入 uint8 格式。")
                            img_input = img_input.astype(np.uint8)
                    img = Image.fromarray(img_input, 'RGB')
                elif isinstance(img_input, Image.Image):
                    img = img_input.convert('RGB')
                else:
                    logging.warning(f"列表中发现不支持的输入类型 {type(img_input)}，将跳过此项。")
                    continue

                img_tensor = self.transform(img)
                processed_tensors.append(img_tensor)
                valid_indices.append(i)

            except FileNotFoundError as e:
                logging.error(f"{e} (索引 {i})。跳过此图像。")
            except Exception as e:
                logging.error(f"处理单个图像 (索引 {i}) 时出错: {e}。跳过此图像。")

        if not processed_tensors:
            logging.warning("输入列表中的所有图像都未能成功处理。")
            return np.empty((0, self.feature_dim), dtype=np.float32)

        # 2. 批处理和模型推理
        batch_tensor = torch.stack(processed_tensors, dim=0).to(self.device)
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            output_features_tensor = outputs.last_hidden_state[:, 0, :]

        # 3. 转换为 NumPy 数组
        output_features_np = output_features_tensor.cpu().numpy()

        # 4. L2 归一化
        if normalize:
            # logging.info("对输出特征向量进行 L2 归一化。")
            norms = np.linalg.norm(output_features_np, axis=1, keepdims=True)
            epsilon = 1e-12
            output_features_np = output_features_np / (norms + epsilon)

        # 5. 处理部分失败的情况并返回结果
        if len(output_features_np) != len(imgs):
            final_output = np.full((len(imgs), self.feature_dim), np.nan, dtype=np.float32)
            for i, feature_vec in zip(valid_indices, output_features_np):
                final_output[i] = feature_vec
            return final_output
        else:
            return output_features_np
