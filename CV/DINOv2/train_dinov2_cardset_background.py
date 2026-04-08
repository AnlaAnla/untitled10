import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import cv2
import math
import torch.nn.functional as F
import pickle
from transformers import Dinov2Model, Dinov2Config

# ================= 配置区域 =================

TRAIN_DIR = r"/home/martin/ML/Image/CardCls/panini_archive_resize392_dinov2_background/train"
SAVE_DIR = r"/home/martin/ML/Model/card_retrieval/dinov2_base_392_CardSetBackground01"

# 之前如果卡住了，可能生成了坏的缓存文件，建议手动删掉 .pkl 再跑，或者改个名字
RESUME_FROM = os.path.join(SAVE_DIR, "last_checkpoint.pth")
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pth")
BACKBONE_ONLY_PATH = os.path.join(SAVE_DIR, "dinov2_backbone_bg_768.pth")

# 使用 HuggingFace 的模型 ID
HF_MODEL_ID = "facebook/dinov2-base"

IMG_SIZE = 392
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 3e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= 1. 数据集 =================
class CardTypeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        cache_path = os.path.join(os.path.dirname(root_dir), "dataset_cache_bg_type.pkl")

        if os.path.exists(cache_path):
            print(f"Loading cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                self.samples = data['samples']
                self.class_map = data['class_map']
        else:
            print("Building Index for Card Types...")
            self.class_map = {}
            self.samples = []

            subfolders = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])

            for folder_name in tqdm(subfolders):
                if folder_name not in self.class_map:
                    self.class_map[folder_name] = len(self.class_map)

                label_idx = self.class_map[folder_name]
                folder_path = os.path.join(root_dir, folder_name)
                for entry in os.scandir(folder_path):
                    if entry.name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        self.samples.append((entry.path, label_idx))

            print(f"Index built! Found {len(self.class_map)} types, {len(self.samples)} images.")
            with open(cache_path, 'wb') as f:
                pickle.dump({'samples': self.samples, 'class_map': self.class_map}, f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            # 遇到坏图返回纯黑，防止训练中断
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)['image']
        return img, label


# ================= 2. ArcFace & Model (HF Version) =================
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2).clamp(0, 1))
        phi = cosine * math.cos(self.m) - sine * math.sin(self.m)
        one_hot = torch.zeros_like(cosine).scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


class Dinov2TypeModel(nn.Module):
    def __init__(self, num_classes, freeze_blocks=8):
        super().__init__()
        print(f"Loading HuggingFace Model: {HF_MODEL_ID}...")

        # 使用 Transformers 加载，这会自动走 hf-mirror
        self.backbone = Dinov2Model.from_pretrained(HF_MODEL_ID)

        # 获取 Embedding 维度 (Base通常是768)
        dim = self.backbone.config.hidden_size

        # --- 冻结策略 ---
        # 冻结所有参数
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 解冻最后几层 Layer (DinoV2-Base 有 12 层 encoder.layer)
        # HuggingFace 的层命名结构是 backbone.encoder.layer[i]
        total_layers = len(self.backbone.encoder.layer)
        print(f"Model has {total_layers} layers. Freezing first {freeze_blocks}.")

        for i in range(freeze_blocks, total_layers):
            for param in self.backbone.encoder.layer[i].parameters():
                param.requires_grad = True

        # 解冻 Layernorm
        for param in self.backbone.layernorm.parameters():
            param.requires_grad = True

        self.bn_neck = nn.BatchNorm1d(dim)
        self.head = ArcMarginProduct(dim, num_classes)

    def forward(self, x, label=None):
        # HF 模型的输出是一个对象
        outputs = self.backbone(x)

        # 取出 CLS token (Batch, 1, 768) -> (Batch, 768)
        # DinoV2 的 CLS token 通常在序列的第一个位置
        cls_token = outputs.last_hidden_state[:, 0, :]

        feat = self.bn_neck(cls_token)

        if label is not None:
            return self.head(feat, label)
        return F.normalize(feat, p=2, dim=1)


# ================= 3. 主训练流程 =================
def main():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

    # --- 修复 Albumentations 警告 ---
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_CUBIC),
        A.HorizontalFlip(p=0.5),

        # 修复: Solarize 现在通常只需要 p (threshold 默认 128)
        # 或者使用 threshold=(128, 128) 明确指定范围
        A.Solarize(threshold=(128, 128), p=0.2),

        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),

        # 修复: Rotate 移除了 value 参数，改用 cval 且只有在 border_mode 为 Constant 时有效
        # 这里的 SafeRotate 或者简单 Rotate 即可
        A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.3),

        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    dataset = CardTypeDataset(TRAIN_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    num_classes = len(dataset.class_map)
    print(f"Total Card Types (Classes): {num_classes}")

    model = Dinov2TypeModel(num_classes).to(DEVICE)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_loss = float('inf')

    if os.path.exists(RESUME_FROM):
        print(f"Loading checkpoint: {RESUME_FROM}")
        ckpt = torch.load(RESUME_FROM, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))

    print("Start Training Background/Texture Model (HF Version)...")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        running_loss = 0.0
        correct = 0
        total_samples = 0

        for i, (imgs, labels) in enumerate(pbar):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                logits = model(imgs, labels)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                correct += (pred == labels).sum().item()
                total_samples += labels.size(0)

            running_loss += loss.item()
            curr_loss = running_loss / (i + 1)
            curr_acc = correct / total_samples

            pbar.set_postfix({'loss': f"{curr_loss:.4f}", 'acc': f"{curr_acc:.2%}"})

        avg_epoch_loss = running_loss / len(dataloader)

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(), 'loss': best_loss,
            }, BEST_MODEL_PATH)
            print(f"--- Best Model Saved (Loss: {best_loss:.4f}) ---")

        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'best_loss': best_loss
        }, RESUME_FROM)

        # 保存 Backbone (保存成 HF 格式 或者 纯参数字典)
        # 这里为了配合你的 MyBatchModel_DINOV2.py，我们最好保存成 huggingface 格式
        # 或者简单点，保存 state_dict，你在推理时手动加载
        raw_model = model.module if hasattr(model, "module") else model
        inf_dict = {k: v for k, v in raw_model.state_dict().items() if "head" not in k}
        torch.save(inf_dict, BACKBONE_ONLY_PATH)


if __name__ == '__main__':
    main()