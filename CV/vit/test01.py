import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from transformers import ViTModel
from sklearn.metrics import accuracy_score

# ================= 配置区域 =================
DATA_DIR = r"/home/martin/ML/Image/CardCls/pokemon_tc_us"
TRAIN_DIR = os.path.join(DATA_DIR, "train_resized")
SAVE_MODEL_PATH = "/home/martin/ML/Model/pokemon_cls/vit-base-patch16-224-Pokemon03"

# 恢复路径
RESUME_PATH = "best_model.pth"
# RESUME_PATH = None

# 【核心修改 2】Batch Size 调整
IMAGE_SIZE = 224
# 单张卡的 Batch Size。V100 16GB 跑 64 比较稳。
# 双卡并行时，物理总 Batch = 64 * 2 = 128
BATCH_SIZE = 64

# 梯度累积步数。
# 等效 Batch Size = 128 * 2 = 256 (满足你的需求)
GRAD_ACCUMULATION_STEPS = 2

EPOCHS = 180
WEIGHT_DECAY = 0.05
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= 模型定义 =================
class MetricViT(nn.Module):
    def __init__(self, num_classes, pretrained_name='google/vit-base-patch16-224'):
        super().__init__()
        # 使用 local_files_only 防止网络报错
        self.vit = ViTModel.from_pretrained(pretrained_name, local_files_only=True)
        self.hidden_dim = self.vit.config.hidden_size
        self.classifier = nn.Linear(self.hidden_dim, num_classes, bias=False)

    def forward(self, x, labels=None):
        outputs = self.vit(x)
        cls_token = outputs.last_hidden_state[:, 0]
        # L2 归一化，这对 Metric Learning 至关重要
        features = F.normalize(cls_token, p=2, dim=1)
        logits = self.classifier(features) * 30.0
        return logits, features


# ================= 数据增强 =================
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# ================= 主函数 =================
def main():
    print(f"当前使用设备: {DEVICE}")
    print(f"可见显卡数量: {torch.cuda.device_count()} (应该显示 2)")


    # 1. 准备数据
    print("正在扫描数据集目录...")
    full_dataset = datasets.ImageFolder(TRAIN_DIR, transform=data_transforms['train'])
    num_classes = len(full_dataset.classes)
    print(f"检测到 {num_classes} 个类别")

    # 2. 初始化模型结构
    print("正在初始化模型...")
    # 请确保这个路径是你已经下载好的本地路径
    local_model_path = r"/home/martin/.cache/huggingface/hub/models--google--vit-base-patch16-224/snapshots/3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3"

    model = MetricViT(num_classes=num_classes, pretrained_name=local_model_path)

    print(model)

    print("_"*30)
    print(model.vit)
    print("_"*30)
    print(model.classifier)

main()