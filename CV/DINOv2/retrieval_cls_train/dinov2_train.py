import os
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

# ================= 配置区域 =================
TRAIN_DIR = r"/home/martin/ML/Image/CardCls/panini_archive_resize392_dinov2/train"
SAVE_DIR = r"/home/martin/ML/Model/card_retrieval/dinov2_base_392-CardSet01"
RESUME_FROM = "./last_checkpoint.pth"

BEST_MODEL_PATH = os.path.join("best_model.pth")
BACKBONE_ONLY_PATH = os.path.join(SAVE_DIR, "dinov2_backbone_768.pth")

MODEL_TYPE = 'dinov2_vitb14'
IMG_SIZE = 392

BATCH_SIZE = 32
EPOCHS = 16
LEARNING_RATE = 3e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= 1. 数据集 (保持之前的缓存逻辑) =================
class HierarchicalCardDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        cache_path = os.path.join(os.path.dirname(root_dir), "dataset_cache_dinov2_native.pkl")

        if os.path.exists(cache_path):
            print(f"Loading cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                self.samples, self.fine_map, self.coarse_map = data['samples'], data['fine_map'], data['coarse_map']
        else:
            print("Building Index for 130k images...")
            self.fine_map, self.coarse_map = {}, {}
            self.samples = []
            subfolders = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])
            for folder_name in tqdm(subfolders):
                if folder_name not in self.fine_map: self.fine_map[folder_name] = len(self.fine_map)
                f_lbl = self.fine_map[folder_name]
                tags = folder_name.split(", ")
                c_key = f"{tags[0]}_{tags[1]}_{tags[3]}_{tags[4]}" if len(tags) >= 5 else folder_name
                if c_key not in self.coarse_map: self.coarse_map[c_key] = len(self.coarse_map)
                c_lbl = self.coarse_map[c_key]
                for entry in os.scandir(os.path.join(root_dir, folder_name)):
                    if entry.name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        self.samples.append((entry.path, f_lbl, c_lbl))
            with open(cache_path, 'wb') as f:
                pickle.dump({'samples': self.samples, 'fine_map': self.fine_map, 'coarse_map': self.coarse_map}, f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, f, c = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform: img = self.transform(image=img)['image']
        return img, f, c


# ================= 2. ArcFace & Model =================
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s, self.m = s, m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2).clamp(0, 1))
        phi = cosine * math.cos(self.m) - sine * math.sin(self.m)
        one_hot = torch.zeros_like(cosine).scatter_(1, label.view(-1, 1).long(), 1)
        return ((one_hot * phi) + ((1.0 - one_hot) * cosine)) * self.s


class Dinov2NativeModel(nn.Module):
    def __init__(self, num_fine, num_coarse, freeze_blocks=8):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        dim = self.backbone.embed_dim

        for param in self.backbone.parameters(): param.requires_grad = False
        for i in range(freeze_blocks, 12):
            for param in self.backbone.blocks[i].parameters(): param.requires_grad = True
        for param in self.backbone.norm.parameters(): param.requires_grad = True

        self.bn_neck = nn.BatchNorm1d(dim)
        self.head_fine = ArcMarginProduct(dim, num_fine)
        self.head_coarse = nn.Linear(dim, num_coarse)

    def forward(self, x, f_lbl=None):
        feat = self.backbone(x)
        feat = self.bn_neck(feat)
        if f_lbl is not None:
            return self.head_fine(feat, f_lbl), self.head_coarse(feat)
        return F.normalize(feat, p=2, dim=1)


# ================= 3. 主训练流程 =================
def main():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_CUBIC),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    dataset = HierarchicalCardDataset(TRAIN_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10, pin_memory=True)

    model = Dinov2NativeModel(len(dataset.fine_map), len(dataset.coarse_map)).to(DEVICE)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    # 解决 DeprecationWarning: 使用新的 amp API
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_loss = float('inf')  # 新增：初始化最高 Loss 为无穷大

    if os.path.exists(RESUME_FROM):
        print(f"Loading checkpoint: {RESUME_FROM}")
        ckpt = torch.load(RESUME_FROM, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))  # 新增：从断点恢复 best_loss

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

        # 统计指标
        running_loss = 0.0
        correct_f = 0
        correct_c = 0
        total_samples = 0

        for i, (imgs, f_lbl, c_lbl) in enumerate(pbar):
            imgs, f_lbl, c_lbl = imgs.to(DEVICE), f_lbl.to(DEVICE), c_lbl.to(DEVICE)

            optimizer.zero_grad()

            # 解决 DeprecationWarning: 使用新的 autocast API
            with torch.amp.autocast('cuda'):
                l_f, l_c = model(imgs, f_lbl)
                loss_f = criterion(l_f, f_lbl)
                loss_c = criterion(l_c, c_lbl)
                loss = loss_f + 0.3 * loss_c

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 计算准确率
            with torch.no_grad():
                pred_f = torch.argmax(l_f, dim=1)
                pred_c = torch.argmax(l_c, dim=1)
                correct_f += (pred_f == f_lbl).sum().item()
                correct_c += (pred_c == c_lbl).sum().item()
                total_samples += f_lbl.size(0)

            # 更新进度条展示
            running_loss += loss.item()
            curr_loss = running_loss / (i + 1)
            curr_acc_f = correct_f / total_samples
            curr_acc_c = correct_c / total_samples

            pbar.set_postfix({
                'loss': f"{curr_loss:.3f}",
                'acc_SKU': f"{curr_acc_f:.2%}",
                'acc_SPU': f"{curr_acc_c:.2%}"
            })

        # 训练结束后计算平均 Loss
        avg_epoch_loss = running_loss / len(dataloader)

        # 新增：保存 loss 最低的模型
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': best_loss,
            }, BEST_MODEL_PATH)
            print(f"--- Best Model Saved (Loss: {best_loss:.4f}) ---")

        # 保存完整检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss  # 新增
        }, RESUME_FROM)

        # 保存纯净推理权重
        raw_model = model.module if hasattr(model, "module") else model
        inf_dict = {k: v for k, v in raw_model.state_dict().items() if "head" not in k}
        torch.save(inf_dict, BACKBONE_ONLY_PATH)


if __name__ == '__main__':
    main()
