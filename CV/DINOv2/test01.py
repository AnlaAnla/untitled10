import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from PIL import Image
from transformers import Dinov2Model
from safetensors.torch import load_file
import torchvision.transforms as transforms
import os

# ================= 配置区域 =================
YOLO_PATH = r"/home/martin/ML/Model/card_cls/yolov11n_card_seg01.onnx"
DINO_MODEL_DIR = "dinov2_hf_local"  # 你转换后的 HF 模型目录
IMG_SIZE = 392  # 训练时的尺寸


# ================= 1. 内存版 Letterbox Resize (完全复刻训练逻辑) =================
def letterbox_image(img_bgr, target_size=392):
    """
    将 OpenCV 读取的图片(BGR) 进行 Letterbox 处理，不保存文件，直接返回 RGB Numpy
    """
    shape = img_bgr.shape[:2]  # [h, w]
    new_shape = (target_size, target_size)

    # 计算缩放比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # 计算新尺寸
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    # 计算 Padding
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    # 1. Resize (使用 CUBIC，保持和训练一致)
    if shape[::-1] != new_unpad:
        img_bgr = cv2.resize(img_bgr, new_unpad, interpolation=cv2.INTER_CUBIC)

    # 2. Padding (补黑边)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img_final = cv2.copyMakeBorder(img_bgr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # 3. 再次 Resize 确保万无一失 (防止舍入误差导致差1像素)
    img_final = cv2.resize(img_final, new_shape, interpolation=cv2.INTER_CUBIC)

    # 4. 转 RGB (OpenCV 默认 BGR, DINOv2 需要 RGB)
    img_rgb = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)

    return img_rgb


# ================= 2. 简化的 YOLO 推理类 =================
class CardDetector:
    def __init__(self, model_path):
        # task='segment' 或 'detect' 取决于模型导出时的设定，onnx通常通用
        self.model = YOLO(model_path, task='segment')

    def detect_and_crop(self, img_path):
        """
        读取图片 -> YOLO 推理 -> 裁剪出最大卡片 -> Letterbox 处理
        返回: (Original Image for debug, Processed RGB Image for DINO)
        """
        # 读取原始图片
        img0 = cv2.imread(img_path)
        if img0 is None:
            raise ValueError(f"无法读取图片: {img_path}")

        # YOLO 推理
        results = self.model.predict(img0, imgsz=640, conf=0.25, verbose=False)[0]

        # 寻找最大面积的框
        max_area = 0
        best_crop = img0  # 如果没检测到，默认用原图

        boxes = results.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            for box in xyxy:
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    # 坐标限制在图片范围内
                    h, w = img0.shape[:2]
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                    best_crop = img0[y1:y2, x1:x2]

        # 如果裁剪出的图太小，可能是误检，还是用原图安全
        if best_crop.size == 0:
            best_crop = img0

        # 执行 Letterbox 处理
        processed_img = letterbox_image(best_crop, target_size=IMG_SIZE)

        return best_crop, processed_img


# ================= 3. DINOv2 特征提取类 (适配 HF 格式) =================
class DinoFeatureExtractor:
    def __init__(self, model_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading DINOv2 from {model_dir} on {self.device}...")

        # 加载骨干网络
        self.backbone = Dinov2Model.from_pretrained(model_dir, local_files_only=True).to(self.device)
        self.backbone.eval()

        # 加载 BN Neck (从 safetensors 手动读取)
        # 注意：训练时我们用了 bn_neck，推理如果不用，相似度会下降
        self.bn_neck = torch.nn.BatchNorm1d(768).to(self.device)
        self.bn_neck.eval()

        st_path = os.path.join(model_dir, "model.safetensors")
        self._load_bn_weights(st_path)

        # 预处理 (标准化)
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_bn_weights(self, path):
        try:
            state_dict = load_file(path)
            bn_dict = {}
            for k, v in state_dict.items():
                if k.startswith("bn_neck."):
                    bn_dict[k.replace("bn_neck.", "")] = v
            if bn_dict:
                self.bn_neck.load_state_dict(bn_dict)
                print("✅ BN Neck weights loaded.")
            else:
                print("⚠️ Warning: No BN Neck weights found.")
        except Exception as e:
            print(f"Error loading BN: {e}")

    @torch.no_grad()
    def extract(self, img_rgb_numpy):
        # Numpy (H,W,C) -> Tensor
        img = Image.fromarray(img_rgb_numpy)
        img_t = self.preprocess(img).unsqueeze(0).to(self.device)

        # Backbone 推理
        outputs = self.backbone(img_t)
        cls_token = outputs.last_hidden_state[:, 0, :]  # [1, 768]

        # BN Neck 处理
        feat = self.bn_neck(cls_token)

        # 归一化 (L2 Normalize) -> 使得点积等于余弦相似度
        feat = F.normalize(feat, p=2, dim=1)

        return feat.cpu().numpy()[0]


# ================= 4. 主程序 =================
if __name__ == "__main__":
    # --- 输入两张未处理的原始图片路径 ---
    # 请修改这里的路径为你想要测试的图片
    IMG_A_PATH = "/home/martin/ML/Image/CardCls/panini_archive/2023_Contenders Optic57ce7473-b2e1-4457-8fbc-4f71c646ef7f/2023, Contenders Optic, 1986 Tribute Autographs Blue, 10, Carmelo Anthony, New York Knicks/5fc603e5-3714-4867-a3ff-db09bc9a3c8d.jpg"
    IMG_B_PATH = "/home/martin/ML/Image/CardCls/pokemon_tc_us_Test/train/('1502952', {'tc'}, '臭臭花'), 002/632af6cd-9482-4bc3-8c07-a2a59b4bd74b.png"

    # 1. 初始化模型
    detector = CardDetector(YOLO_PATH)
    extractor = DinoFeatureExtractor(DINO_MODEL_DIR)

    try:
        print("\n--- Processing Image A ---")
        crop_a, input_a = detector.detect_and_crop(IMG_A_PATH)
        feat_a = extractor.extract(input_a)

        print("\n--- Processing Image B ---")
        crop_b, input_b = detector.detect_and_crop(IMG_B_PATH)
        feat_b = extractor.extract(input_b)

        # 2. 计算余弦相似度
        # 因为在 extract 内部已经做了 Normalize，所以直接点积就是余弦相似度
        similarity = np.dot(feat_a, feat_b)

        print("\n================RESULT================")
        print(f"Image A Path: {IMG_A_PATH}")
        print(f"Image B Path: {IMG_B_PATH}")
        print("-" * 30)
        print(f"Cosine Similarity: {similarity:.4f}")
        print("======================================")

        if similarity > 0.85:  # 阈值根据你的业务调整，通常同卡在0.9以上
            print("判定结果: ✅ 是同一张卡 (Same Card)")
        else:
            print("判定结果: ❌ 不是同一张卡 (Different Card)")

        # 3. (可选) 显示处理后的中间图，确认 YOLO 切得对不对
        # cv2.imshow("Crop A", cv2.cvtColor(input_a, cv2.COLOR_RGB2BGR))
        # cv2.imshow("Crop B", cv2.cvtColor(input_b, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    except Exception as e:
        print(f"\n❌ Error detected: {e}")
        # 如果没有图片，创建一个假的测试文件提示
        if not os.path.exists(IMG_A_PATH):
            print(f"提示: 请修改代码中的 IMG_A_PATH 指向真实的图片路径。")