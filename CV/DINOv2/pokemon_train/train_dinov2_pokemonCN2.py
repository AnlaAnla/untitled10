import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from tqdm import tqdm
import cv2
import pickle
import random
from transformers import Dinov2Model

# ================= 配置区域 =================
TRAIN_DIR = r"/home/martin/ML/Image/CardCls/pokemon_cn_resize392_dinov2/train"
SAVE_DIR = r"/home/martin/ML/Model/pokemon_cls/dinov2_contrastive_392"

BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_retrieval_model.pth")

HF_MODEL_ID = "facebook/dinov2-base"

IMG_SIZE = 392
BATCH_SIZE = 32  # 对比学习Batch越大越好，如果显存够可以开64
EPOCHS = 100
LEARNING_RATE = 2e-5
TEMPERATURE = 0.07  # 对比学习核心超参数，通常取 0.05 ~ 0.1
EVAL_INTERVAL = 1  # 每 2 轮进行一次真实的向量检索测试

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= 自定义模拟反光增强 =================
class RandomGlareA(ImageOnlyTransform):
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


# ================= 1. 双视图数据集 (核心变动) =================
# 这个Dataset每次返回两张图：一张纯净的标准图，一张被强力破坏的模拟图
class PairRetrievalDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        cache_path = os.path.join(os.path.dirname(root_dir), "dataset_cache_contrastive.pkl")

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.samples = pickle.load(f)
        else:
            print("Scanning images...")
            self.samples = []
            subfolders = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])
            for label_idx, folder_name in enumerate(tqdm(subfolders)):
                folder_path = os.path.join(root_dir, folder_name)
                for entry in os.scandir(folder_path):
                    if entry.name.lower().endswith(('.jpg', '.png', '.webp')):
                        self.samples.append((entry.path, label_idx))
            with open(cache_path, 'wb') as f:
                pickle.dump(self.samples, f)

        # 视图 1：纯净底库特征 (只做 Resize 和 Normalize)
        self.transform_clean = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        # 视图 2：模拟真实手机拍摄特征 (疯狂增强)
        self.transform_aug = A.Compose([
            # 0. 基础 Resize 保持不变，确保全图都在
            A.Resize(height=IMG_SIZE, width=IMG_SIZE, interpolation=cv2.INTER_CUBIC),

            # 1. 模拟真实透视变形 (幅度调小一点，防止把角拉伸出画面)
            A.Perspective(scale=(0.02, 0.05), p=0.5),

            # 2. 解决发灰、对比度极低的光照环境 (极其关键，解决小拉达变铁蚁的核心！)
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=(-0.4, 0.1), p=0.6),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),

            # 3. 模拟漫反射大面积反光白雾 (兼容新旧版本写法)
            A.RandomSunFlare(src_radius=100, p=0.3),

            # 4. 原有的条状物理反光
            RandomGlareA(p=0.4),

            # 5. 模拟手机镜头模糊和噪点
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=5, p=1.0),
            ], p=0.3),
            A.GaussNoise(std_range=(0.1, 0.3), p=0.4),

            # 6. 模拟手指遮挡 (把最大遮挡块调小一点，防止刚好把左下角全部捂住)
            A.CoarseDropout(num_holes_range=(2, 4), hole_height_range=(16, 32), hole_width_range=(16, 32), fill=0,
                            p=0.4),

            # 7. YOLO 抠图带来的轻微旋转误差
            A.SafeRotate(limit=5, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5),

            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros((IMG_SIZE, IMG_SIZE, 3),
                                                                                    dtype=np.uint8)

        # 产生两个视图
        img_clean = self.transform_clean(image=img)['image']
        img_aug = self.transform_aug(image=img)['image']

        return img_clean, img_aug, label


# ================= 2. 极简向量检索模型 =================
class Dinov2Retrieval(nn.Module):
    def __init__(self, freeze_blocks=6):
        super().__init__()
        print("Loading DINOv2 Backbone...")
        self.backbone = Dinov2Model.from_pretrained(HF_MODEL_ID)

        for param in self.backbone.parameters():
            param.requires_grad = False

        # 解冻后半部分
        total_layers = len(self.backbone.encoder.layer)
        for i in range(freeze_blocks, total_layers):
            for param in self.backbone.encoder.layer[i].parameters():
                param.requires_grad = True
        for param in self.backbone.layernorm.parameters():
            param.requires_grad = True

    def forward(self, x):
        outputs = self.backbone(x)
        cls_token = outputs.last_hidden_state[:, 0, :]
        # 抛弃复杂的 Head，直接 L2 归一化输出向量！
        return F.normalize(cls_token, p=2, dim=1)


# ================= 3. InfoNCE 对比损失函数 =================
def contrastive_loss(feat_clean, feat_aug, temperature=TEMPERATURE):
    """
    拉近 feat_clean 和 feat_aug，推开 batch 内的其他图片
    """
    batch_size = feat_clean.size(0)

    # 计算 Clean 和 Aug 之间的相似度矩阵 [Batch, Batch]
    logits = torch.matmul(feat_clean, feat_aug.T) / temperature

    # 对角线上的元素是正样本 (同一个图像的 Clean 和 Aug)
    labels = torch.arange(batch_size).to(feat_clean.device)

    # 分别从两个方向计算 CrossEntropy
    loss_1 = F.cross_entropy(logits, labels)
    loss_2 = F.cross_entropy(logits.T, labels)

    return (loss_1 + loss_2) / 2.0


# ================= 4. 主流程 =================
def main():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

    dataset = PairRetrievalDataset(TRAIN_DIR)
    # drop_last=True 防止对比学习最后一个batch过小导致loss不稳定
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True,
                            drop_last=True)

    model = Dinov2Retrieval().to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    best_top1_acc = 0.0

    print("🚀 开始对比学习检索训练...")

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")
        running_loss = 0.0

        for i, (imgs_clean, imgs_aug, _) in enumerate(pbar):
            imgs_clean = imgs_clean.to(DEVICE)
            imgs_aug = imgs_aug.to(DEVICE)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                feat_clean = model(imgs_clean)
                feat_aug = model(imgs_aug)
                loss = contrastive_loss(feat_clean, feat_aug, TEMPERATURE)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({'InfoNCE Loss': f"{running_loss / (i + 1):.4f}"})

        # ==================================================
        # 模拟真实线上环境的验证逻辑：建库 -> 检索 -> 算准确率
        # ==================================================
        if (epoch + 1) % EVAL_INTERVAL == 0:
            model.eval()
            print("\n🔍 正在进行检索准确率测试...")

            gallery_feats = []
            gallery_labels = []

            query_feats = []
            query_labels = []

            with torch.no_grad():
                # 抽取一部分数据做测试 (比如前 500 个 batch)
                eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

                for k, (imgs_c, imgs_a, lbls) in enumerate(tqdm(eval_loader, desc="Extracting Features", leave=False)):
                    imgs_c, imgs_a = imgs_c.to(DEVICE), imgs_a.to(DEVICE)

                    with torch.amp.autocast('cuda'):
                        f_c = model(imgs_c)  # 干净图当底库
                        f_a = model(imgs_a)  # 增强图当用户的查询图

                    gallery_feats.append(f_c.cpu())
                    gallery_labels.append(lbls)

                    query_feats.append(f_a.cpu())
                    query_labels.append(lbls)

                    if k > 100: break  # 测试大概 3000 张图即可，没必要全跑节省时间

            # 拼接向量张量
            gallery_feats = torch.cat(gallery_feats, dim=0)  # [N, 768]
            gallery_labels = torch.cat(gallery_labels, dim=0)  # [N]

            query_feats = torch.cat(query_feats, dim=0)  # [N, 768]
            query_labels = torch.cat(query_labels, dim=0)  # [N]

            # 计算相似度矩阵 [Query_N, Gallery_N]
            sim_matrix = torch.matmul(query_feats, gallery_feats.T)

            # 取 Top-1 和 Top-5
            top5_idx = sim_matrix.topk(5, dim=1).indices

            correct_1 = 0
            correct_5 = 0
            total = query_labels.size(0)

            for idx in range(total):
                q_label = query_labels[idx]
                retrieved_labels = gallery_labels[top5_idx[idx]]

                if q_label == retrieved_labels[0]:
                    correct_1 += 1
                if q_label in retrieved_labels:
                    correct_5 += 1

            top1_acc = correct_1 / total
            top5_acc = correct_5 / total

            print(f"📊 [Epoch {epoch + 1} Eval] Retrieval Top-1: {top1_acc:.2%} | Top-5: {top5_acc:.2%}")

            # 保存最佳模型
            if top1_acc > best_top1_acc:
                best_top1_acc = top1_acc
                # 因为模型结构变了（去掉了Head），可以直接保存完整的 state_dict，你的导出脚本直接读它即可！
                raw_model = model.module if hasattr(model, "module") else model
                torch.save(raw_model.state_dict(), BEST_MODEL_PATH)
                print(f"✅ 保存新的最佳检索模型! Top-1: {top1_acc:.2%}")


if __name__ == '__main__':
    main()