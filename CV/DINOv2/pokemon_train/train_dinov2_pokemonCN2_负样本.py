import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, BatchSampler
from collections import defaultdict
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

        # 注意：这里我故意换了一个新的 cache 文件名，避免你旧缓存格式不兼容
        cache_path = os.path.join(os.path.dirname(root_dir), "dataset_cache_contrastive_hardneg_v2.pkl")

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.samples = pickle.load(f)
        else:
            print("Scanning images...")
            self.samples = []
            subfolders = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])

            for label_idx, folder_name in enumerate(tqdm(subfolders)):
                folder_path = os.path.join(root_dir, folder_name)
                group_key = self.parse_group_key(folder_name)

                for entry in os.scandir(folder_path):
                    if entry.name.lower().endswith(('.jpg', '.png', '.webp')):
                        self.samples.append((entry.path, label_idx, group_key))

            with open(cache_path, 'wb') as f:
                pickle.dump(self.samples, f)

        # 建立 group -> indices 的映射，给 BatchSampler 用
        self.group_to_indices = defaultdict(list)
        for idx, (_, label, group_key) in enumerate(self.samples):
            self.group_to_indices[group_key].append(idx)

        # 只保留那些至少有 2 个样本的 group，才有资格做硬负样本组
        self.valid_hard_groups = [
            g for g, idxs in self.group_to_indices.items()
            if len(idxs) >= 2
        ]

        print(f"Total samples: {len(self.samples)}")
        print(f"Hard-negative groups (>=2 samples): {len(self.valid_hard_groups)}")

        # 视图 1：纯净底库特征
        self.transform_clean = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        # 视图 2：模拟真实手机拍摄特征
        self.transform_aug = A.Compose([
            A.Resize(height=IMG_SIZE, width=IMG_SIZE, interpolation=cv2.INTER_CUBIC),

            A.Perspective(scale=(0.02, 0.05), p=0.5),

            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=(-0.4, 0.1), p=0.6),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),

            A.RandomSunFlare(src_radius=100, p=0.3),

            RandomGlareA(p=0.4),

            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=5, p=1.0),
            ], p=0.3),
            A.GaussNoise(std_range=(0.1, 0.3), p=0.4),

            A.CoarseDropout(
                num_holes_range=(2, 4),
                hole_height_range=(16, 32),
                hole_width_range=(16, 32),
                fill=0,
                p=0.4
            ),

            A.SafeRotate(limit=5, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5),

            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    @staticmethod
    def parse_group_key(folder_name: str) -> str:
        """
        文件夹名示例：
        14531,狠辣椒,012_129,Master Ball Reverse Holo

        我们取：
        狠辣椒,012_129
        """
        parts = [x.strip() for x in folder_name.split(",")]

        if len(parts) >= 3:
            card_name = parts[1]
            card_no = parts[2]
            return f"{card_name},{card_no}"

        # 兜底，防止极少数脏数据命名不规范
        return folder_name

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, group_key = self.samples[idx]

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros(
            (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8
        )

        img_clean = self.transform_clean(image=img)['image']
        img_aug = self.transform_aug(image=img)['image']

        return img_clean, img_aug, label


class HardNegativeBatchSampler(BatchSampler):
    """
    目标：
    - 大多数 batch 正常随机
    - 有一定概率构造“硬负样本 batch”
    - 硬负样本 batch 里，会先塞入一部分同 group_key 的样本
      例如同属 “狠辣椒,012_129”
    - 然后再用随机样本补满整个 batch

    这样这些“同主结构、细节不同”的卡，就会在 InfoNCE 里自动成为更难的负样本
    """
    def __init__(
        self,
        dataset,
        batch_size,
        hard_negative_prob=0.35,
        hard_group_size=8,
        drop_last=True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.hard_negative_prob = hard_negative_prob
        self.hard_group_size = min(hard_group_size, batch_size)
        self.drop_last = drop_last

        self.num_samples = len(dataset)
        self.all_indices = list(range(self.num_samples))

    def __iter__(self):
        unused = set(self.all_indices)

        while len(unused) >= self.batch_size:
            batch = []

            # 1) 先尝试构造一个“硬负样本子块”
            if (
                len(self.dataset.valid_hard_groups) > 0 and
                random.random() < self.hard_negative_prob
            ):
                # 找还有足够剩余样本的 group
                candidate_groups = []
                for g in self.dataset.valid_hard_groups:
                    available = [i for i in self.dataset.group_to_indices[g] if i in unused]
                    if len(available) >= 2:
                        candidate_groups.append((g, available))

                if len(candidate_groups) > 0:
                    g, available = random.choice(candidate_groups)
                    take_n = min(len(available), self.hard_group_size, self.batch_size)

                    chosen_hard = random.sample(available, take_n)
                    batch.extend(chosen_hard)

                    for i in chosen_hard:
                        unused.remove(i)

            # 2) 剩余位置用随机样本补齐
            remain = self.batch_size - len(batch)
            if remain > 0:
                if len(unused) < remain:
                    if self.drop_last:
                        break
                    else:
                        batch.extend(list(unused))
                        unused.clear()
                else:
                    chosen_rand = random.sample(list(unused), remain)
                    batch.extend(chosen_rand)
                    for i in chosen_rand:
                        unused.remove(i)

            if len(batch) == self.batch_size:
                random.shuffle(batch)
                yield batch

        # 不足一个 batch 的尾巴，按 drop_last 处理
        if not self.drop_last and len(unused) > 0:
            yield list(unused)

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size


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

    batch_sampler = HardNegativeBatchSampler(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        hard_negative_prob=0.35,  # 35% 的 batch 会带一点同 group 样本
        hard_group_size=8,  # 每个 hard batch 先塞 8 张相似卡，再补随机卡
        drop_last=True
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=8,
        pin_memory=True
    )

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