import os

# ================= 0. 国内加速配置 =================
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
from transformers import Dinov2Model

# ================= 配置区域 =================

# 1. 指向【文字/图标类】数据集路径
TRAIN_DIR = r"/home/martin/ML/Image/CardCls/panini_archive_resize392_dinov2_text_icon/train"

# 2. 保存路径 (区分于背景模型)
SAVE_DIR = r"/home/martin/ML/Model/card_retrieval/dinov2_base_392_CardSetTextIcon01"

RESUME_FROM = os.path.join(SAVE_DIR, "last_checkpoint.pth")
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pth")
# 这个模型只用于看文字细节
BACKBONE_ONLY_PATH = os.path.join(SAVE_DIR, "dinov2_backbone_text_768.pth")

HF_MODEL_ID = "facebook/dinov2-base"

IMG_SIZE = 392
BATCH_SIZE = 32
EPOCHS = 25  # 细节学习可能需要更多轮次，但先定20
LEARNING_RATE = 2e-5  # 比背景类稍微低一点，防止破坏微小的特征
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= 1. 数据集 =================
class CardTypeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # 缓存文件名改为 dataset_cache_text_type.pkl
        cache_path = os.path.join(os.path.dirname(root_dir), "dataset_cache_text_type.pkl")

        if os.path.exists(cache_path):
            print(f"Loading cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                self.samples = data['samples']
                self.class_map = data['class_map']
        else:
            print("Building Index for Text/Icon Types...")
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
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)['image']
        return img, label


# ================= 2. ArcFace & Model =================
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
    def __init__(self, num_classes, freeze_blocks=6):  # 这里解冻更多层！
        super().__init__()
        print(f"Loading HuggingFace Model: {HF_MODEL_ID}...")
        self.backbone = Dinov2Model.from_pretrained(HF_MODEL_ID)
        dim = self.backbone.config.hidden_size

        # 冻结策略调整：文字细节需要更深层的特征微调
        # 这里建议只冻结前 6 层，让后 6 层都参与训练
        for param in self.backbone.parameters():
            param.requires_grad = False

        total_layers = len(self.backbone.encoder.layer)
        print(f"Model has {total_layers} layers. Freezing first {freeze_blocks}.")

        for i in range(freeze_blocks, total_layers):
            for param in self.backbone.encoder.layer[i].parameters():
                param.requires_grad = True

        for param in self.backbone.layernorm.parameters():
            param.requires_grad = True

        self.bn_neck = nn.BatchNorm1d(dim)
        self.head = ArcMarginProduct(dim, num_classes)

    def forward(self, x, label=None):
        outputs = self.backbone(x)
        cls_token = outputs.last_hidden_state[:, 0, :]
        feat = self.bn_neck(cls_token)

        if label is not None:
            return self.head(feat, label)
        return F.normalize(feat, p=2, dim=1)


# ================= 3. 主训练流程 =================
def main():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

    # --- 【关键】文字/图标专用增强策略 ---
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_CUBIC),
        A.HorizontalFlip(p=0.5),

        # 1. 锐化 (Sharpen): 突出文字边缘，非常重要
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),

        # 2. 随机遮挡 (CoarseDropout):
        # 挖掉 2-8 个洞，每个洞最大 32x32。
        # 强迫模型不看人脸，去看没被遮挡的角落里的字。
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32,
                        min_holes=2, min_height=16, min_width=16,
                        fill_value=0, p=0.3),

        # 3. 亮度对比度: 模拟金属烫金字在不同光线下的亮度
        # 限制比背景类小一点，防止文字颜色过曝
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),

        # 4. 透视变换 (Perspective):
        # 文字对角度很敏感，稍微扭一下模拟扫描不正
        A.Perspective(scale=(0.05, 0.1), p=0.3),

        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    dataset = CardTypeDataset(TRAIN_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    num_classes = len(dataset.class_map)
    print(f"Total Text/Icon Types: {num_classes}")

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

    print("Start Training Text/Icon Model...")

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

        raw_model = model.module if hasattr(model, "module") else model
        inf_dict = {k: v for k, v in raw_model.state_dict().items() if "head" not in k}
        torch.save(inf_dict, BACKBONE_ONLY_PATH)


if __name__ == '__main__':
    main()