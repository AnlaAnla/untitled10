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
from albumentations.core.transforms_interface import ImageOnlyTransform
from tqdm import tqdm
import cv2
import math
import torch.nn.functional as F
import pickle
import random
from transformers import Dinov2Model

# ================= 配置区域 =================

# 1. 数据集路径
TRAIN_DIR = r"/home/martin/ML/Image/CardCls/pokemon_cn_resize392_dinov2/train"

# 2. 保存路径
SAVE_DIR = r"/home/martin/ML/Model/pokemon_cls/dinov2_base_392_PokemonCN01"

RESUME_FROM = os.path.join(SAVE_DIR, "last_checkpoint.pth")
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pth")
# 这个模型只用于看文字细节
BACKBONE_ONLY_PATH = os.path.join(SAVE_DIR, "dinov2_backbone_pokemon_768.pth")

HF_MODEL_ID = "facebook/dinov2-base"

IMG_SIZE = 392
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 新增：验证测试控制参数 ---
DO_EVAL = True  # 是否使用无增强原图进行准确率验证
EVAL_INTERVAL = 1  #


# ================= 自定义模拟反光增强 (兼容 Albumentations) =================
class RandomGlareA(ImageOnlyTransform):
    """Albumentations 版本的模拟卡牌物理反光/高光带"""

    def __init__(self, always_apply=False, p=0.5):
        super(RandomGlareA, self).__init__(always_apply, p)

    def apply(self, img, **params):
        h, w = img.shape[:2]
        overlay = np.zeros((h, w, 4), dtype=np.uint8)

        x1 = random.randint(0, w // 2)
        y1 = 0
        x2 = random.randint(w // 2, w)
        y2 = h
        width = random.randint(20, h // 3)
        alpha = random.randint(50, 150)

        cv2.line(overlay, (x1, y1), (x2, y2), (255, 255, 255, alpha), thickness=width)
        img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

        mask = overlay[:, :, 3] / 255.0
        for c in range(0, 3):
            img_rgba[:, :, c] = (1. - mask) * img_rgba[:, :, c] + mask * overlay[:, :, c]

        return cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)


# ================= 1. 数据集 =================
class CardTypeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
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

        # 【修复关键 1】：如果 label 为空，说明是验证/推理阶段，直接返回相似度得分！
        if label is None:
            return cosine * self.s

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2).clamp(0, 1))
        phi = cosine * math.cos(self.m) - sine * math.sin(self.m)
        one_hot = torch.zeros_like(cosine).scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


class Dinov2TypeModel(nn.Module):
    def __init__(self, num_classes, freeze_blocks=6):
        super().__init__()
        print(f"Loading HuggingFace Model: {HF_MODEL_ID}...")
        self.backbone = Dinov2Model.from_pretrained(HF_MODEL_ID)
        dim = self.backbone.config.hidden_size

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
        # 【修复关键 2】：防止 Eval 时 BatchNorm 历史统计量损坏导致特征全毁
        if self.training:
            feat = self.bn_neck(cls_token)
        else:
            # 在验证模式下，强制 BN 层使用当前 Batch 的统计量，或者直接 bypass
            self.bn_neck.train()
            feat = self.bn_neck(cls_token)
            self.bn_neck.eval()

        # 将 label 透传给 head，如果是 None，head 就会输出纯净相似度
        return self.head(feat, label)


# ================= 3. 主训练流程 =================
def main():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

    # --- 1. 训练集强数据增强 (Train Transform) ---
    transform_train = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_CUBIC),
        A.Perspective(scale=(0.02, 0.08), p=0.4),
        RandomGlareA(p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
        A.CoarseDropout(num_holes_range=(2, 6), hole_height_range=(16, 40),
                        hole_width_range=(16, 40), fill=0, p=0.4),

        # 1. 模拟手机对焦不准或手抖
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=5, p=1.0),
        ], p=0.3),

        # 2. 模拟真实光线导致的严重色偏（官方图色彩完美，真实拍摄色温差异巨大）
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),

        # 3. 模拟弱光环境下的噪点（非常关键）
        A.GaussNoise(std_range=(0.1, 0.3), p=0.3),

        # 4. 模拟 YOLO 抠图可能存在的轻微旋转误差
        A.SafeRotate(limit=5, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.4),

        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # --- 2. 验证集无增强 (Clean Val Transform) ---
    transform_val = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # --- 3. 构造 DataLoader ---
    train_dataset = CardTypeDataset(TRAIN_DIR, transform=transform_train)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    if DO_EVAL:
        val_dataset = CardTypeDataset(TRAIN_DIR, transform=transform_val)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    num_classes = len(train_dataset.class_map)
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

    # 尝试加载外部 best_model (如存在)
    if os.path.exists(BEST_MODEL_PATH):
        try:
            print("----尝试临时加载 best model---")
            ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(ckpt['model_state_dict'])
            print("-------!!! 成功加载 best model !!!---")
        except Exception as e:
            print(f"加载 best_model 失败: {e}")

    print(f"Start Training Model... (Total Epochs: {EPOCHS}, Eval Enabled: {DO_EVAL}, Eval Interval: {EVAL_INTERVAL})")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")

        running_loss = 0.0
        correct = 0
        total_samples = 0

        # --- 训练阶段 ---
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

        avg_epoch_loss = running_loss / len(train_dataloader)
        print(f"[Epoch {epoch + 1} Train Result] Loss: {avg_epoch_loss:.4f} | Acc: {correct / total_samples:.2%}")

        # --- 验证阶段 (无增强原图测试) ---
        is_eval_epoch = DO_EVAL and ((epoch + 1) % EVAL_INTERVAL == 0)

        if is_eval_epoch:
            model.eval()
            val_correct = 0
            val_total = 0
            val_running_loss = 0.0

            print(f"--- Running Clean Evaluation (Epoch {epoch + 1}) ---")
            with torch.no_grad():
                for val_imgs, val_labels in tqdm(val_dataloader, desc="Evaluating", leave=False):
                    val_imgs, val_labels = val_imgs.to(DEVICE), val_labels.to(DEVICE)

                    with torch.amp.autocast('cuda'):
                        val_logits = model(val_imgs, label=None)
                        loss = criterion(val_logits, val_labels)

                    val_running_loss += loss.item()
                    val_preds = torch.argmax(val_logits, dim=1)
                    val_correct += (val_preds == val_labels).sum().item()
                    val_total += val_labels.size(0)

            val_acc = val_correct / val_total
            val_loss = val_running_loss / len(val_dataloader)

            print(f"★ [Epoch {epoch + 1} Eval Result] Clean Val Loss: {val_loss:.4f} | Clean Val Acc: {val_acc:.2%} ★")

            # 在 Eval 模式下，基于验证集 Loss 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'loss': best_loss,
                    'val_acc': val_acc
                }, BEST_MODEL_PATH)
                print(f"--- [Saved] New Best Model by Val Loss: {best_loss:.4f} (Val Acc: {val_acc:.2%}) ---")

        else:
            # 如果不开 Eval 或当前 Epoch 不是 Eval 轮次
            if not DO_EVAL:
                # 兼容旧逻辑：如果不跑 Eval，就按照 Train Loss 来保存
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'loss': best_loss,
                    }, BEST_MODEL_PATH)
                    print(f"--- [Saved] Best Model by Train Loss: {best_loss:.4f} ---")

        # --- 保存常规检查点 (Checkpoint) ---
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }, RESUME_FROM)

        # --- 提取并保存 Inference 专用骨干网络 ---
        raw_model = model.module if hasattr(model, "module") else model
        inf_dict = {k: v for k, v in raw_model.state_dict().items() if "head" not in k}
        torch.save(inf_dict, BACKBONE_ONLY_PATH)


if __name__ == '__main__':
    main()