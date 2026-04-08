import torch
from transformers import ViTModel, ViTConfig
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import List, Union
import logging
import os
import torch.nn.functional as F
import torchvision.transforms.functional as TF

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PadToSquare:
    def __init__(self, fill=(128, 128, 128)):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        max_wh = max(w, h)
        pad_w = max_wh - w
        pad_h = max_wh - h
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        return TF.pad(img, padding, fill=self.fill, padding_mode='constant')


class MyViTFeatureExtractor:
    def __init__(self, local_model_path: str) -> None:
        """
        初始化特征提取器。

        适配: 能够加载由 MetricViT 训练并保存的 backbone 模型。
        """
        if not os.path.isdir(local_model_path):
            raise NotADirectoryError(f"Model path not found: {local_model_path}")

        logging.info(f"Loading model from: {local_model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        try:
            # 加载配置和模型 (这里加载的是纯 ViTModel)
            self.config = ViTConfig.from_pretrained(local_model_path, local_files_only=True)
            self.model = ViTModel.from_pretrained(local_model_path, config=self.config, local_files_only=True)
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

        self.model.to(self.device)
        self.model.eval()

        # 定义预处理 (与训练时的 Val Transform 保持一致)
        self.transform = transforms.Compose([
            PadToSquare(fill=(128, 128, 128)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.feature_dim = self.model.config.hidden_size
        logging.info(f"Feature dimension: {self.feature_dim}")

    def run(self, imgs: List[Union[str, np.ndarray, Image.Image]], normalize: bool = True) -> np.ndarray:
        """
        处理一批图像并返回特征向量。

        Args:
            imgs: 图片路径、numpy数组或PIL Image的列表
            normalize: 是否进行 L2 归一化 (强烈建议为 True，适配 Milvus/Cosine 搜索)
        """
        if not imgs:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        processed_tensors = []
        valid_indices = []

        # 1. 预处理
        for i, img_input in enumerate(imgs):
            try:
                # --- 图像读取逻辑 (保持你原有的健壮性逻辑) ---
                if isinstance(img_input, str):
                    img = Image.open(img_input).convert('RGB')
                elif isinstance(img_input, np.ndarray):
                    if img_input.dtype != np.uint8:
                        if img_input.max() <= 1.0:
                            img_input = (img_input * 255).astype(np.uint8)
                        else:
                            img_input = img_input.astype(np.uint8)
                    img = Image.fromarray(img_input, 'RGB')
                elif isinstance(img_input, Image.Image):
                    img = img_input.convert('RGB')
                else:
                    continue
                    # ----------------------------------------

                img_tensor = self.transform(img)
                processed_tensors.append(img_tensor)
                valid_indices.append(i)
            except Exception as e:
                logging.error(f"Error processing image index {i}: {e}")

        if not processed_tensors:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        # 2. 推理
        batch_tensor = torch.stack(processed_tensors, dim=0).to(self.device)

        with torch.no_grad():
            outputs = self.model(batch_tensor)
            # 【关键】提取 last_hidden_state 的 [CLS] token (Index 0)
            # 这与训练时的 MetricViT 保持完全一致
            features = outputs.last_hidden_state[:, 0, :]

        # 3. 后处理
        if normalize:
            # 使用 PyTorch 的 normalize 更精确，或者保持 numpy 实现
            features = F.normalize(features, p=2, dim=1)
            output_np = features.cpu().numpy()
        else:
            output_np = features.cpu().numpy()

        # 4. 填充结果 (保持列表长度一致)
        if len(output_np) != len(imgs):
            final_output = np.full((len(imgs), self.feature_dim), np.nan, dtype=np.float32)
            for idx, vec in zip(valid_indices, output_np):
                final_output[idx] = vec
            return final_output

        return output_np


def compare_images(extractor, img_path_A, img_path_B):
    """
    计算两张图片的相似度
    """
    if not os.path.exists(img_path_A) or not os.path.exists(img_path_B):
        print(f"❌ 错误: 找不到图片路径。\nA: {img_path_A}\nB: {img_path_B}")
        return

    print(f"🔍 正在对比:")
    print(f"  图 A: {os.path.basename(img_path_A)}")
    print(f"  图 B: {os.path.basename(img_path_B)}")

    # 1. 提取特征 (一次传入两张图，效率更高)
    # run 方法返回的是已经归一化过的 numpy 数组
    vectors = extractor.run([img_path_A, img_path_B], normalize=True)

    vec_a = vectors[0]
    vec_b = vectors[1]

    # 2. 计算余弦相似度 (Cosine Similarity)
    # 因为 vec_a 和 vec_b 模长都为 1，所以点积就是余弦相似度
    similarity = np.dot(vec_a, vec_b)

    # 3. 计算欧氏距离 (Euclidean Distance) - 辅助参考
    # 距离越小越相似
    distance = np.linalg.norm(vec_a - vec_b)

    # 4. 打印结果
    print("-" * 30)
    print(f"📊 相似度结果:")
    print(f"  ★ 余弦相似度 (Cosine): {similarity:.4f}  (越接近 1.0 越相似)")
    print(f"  ☆ 欧氏距离 (L2 Dist):  {distance:.4f}    (越接近 0.0 越相似)")
    print("-" * 30)

    # 5. 简单判定建议
    threshold = 0.85  # 这个阈值可以根据实际情况调整
    if similarity > threshold:
        print("✅ 结论: 它们极有可能是同一张卡 (或同一宝可梦的不同语言版本)")
    else:
        print("❌ 结论: 它们看起来是不同的卡片")
    print("\n")


if __name__ == "__main__":
    # ================= 配置 =================
    # 你的模型保存路径
    MODEL_PATH = "/home/martin/ML/Model/pokemon_cls/vit-base-patch16-224-PokemonCN05_best"

    # 这里填入你想测试的两张图片的绝对路径
    # 建议测试：
    # 1. 一张中文卡 vs 同一张的英文卡
    # 2. 一张卡 vs 一张完全不同的卡
    IMG_1 = r"/home/martin/ML/Image/CardCls/pokemon_tc_us_Test/train/('1502952', {'tc'}, '臭臭花'), 002/632af6cd-9482-4bc3-8c07-a2a59b4bd74b.png"
    IMG_2 = r"/home/martin/ML/Image/CardCls/pokemon_tc_us_Test/train/('1502952', {'tc'}, '臭臭花'), 002/632af6cd-9482-4bc3-8c07-a2a59b4bd74b.png"

    # ================= 运行 =================
    try:
        print("正在加载模型，请稍候...")
        # 初始化提取器
        extractor = MyViTFeatureExtractor(MODEL_PATH)

        # 执行对比
        compare_images(extractor, IMG_1, IMG_2)

    except Exception as e:
        print(f"运行出错: {e}")