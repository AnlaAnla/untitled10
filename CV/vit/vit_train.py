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
import torchvision.transforms.functional as TF
import random

# ================= 配置区域 =================
DATA_DIR = r"/home/martin/ML/Image/CardCls/pokemon_cn_224"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
SAVE_MODEL_PATH = "/home/martin/ML/Model/pokemon_cls/vit-base-patch16-224-PokemonCN08"

# 恢复路径
RESUME_PATH = "best_model.pth"
# RESUME_PATH = None

# 单张卡的 Batch Size。V100 16GB 跑 64 比较稳。
# 双卡并行时，物理总 Batch = 64 * 2 = 128
BATCH_SIZE = 64

# 梯度累积步数。
# 等效 Batch Size = 128 * 2 = 256 (满足你的需求)
GRAD_ACCUMULATION_STEPS = 2

EPOCHS = 180
WEIGHT_DECAY = 0.05
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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



class RandomBackgroundPad:
    """以一定概率将卡牌稍微缩小，并放置在随机颜色的纯色背景上，模拟拍照时拍到了桌面的情况"""

    def __init__(self, p=0.5, scale_range=(0.85, 0.95)):
        self.p = p
        self.scale_range = scale_range

    def __call__(self, img):
        if random.random() > self.p:
            return img

        w, h = img.size
        scale = random.uniform(*self.scale_range)
        new_w, new_h = int(w * scale), int(h * scale)

        # 缩小图像
        img_resized = TF.resize(img, (new_h, new_w))

        # 生成随机背景色 (模拟各种颜色的桌面)
        bg_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # 补回原尺寸
        pad_w = w - new_w
        pad_h = h - new_h
        # 随机位置放置 (不一定在正中间，模拟没拍正)
        left = random.randint(0, pad_w)
        top = random.randint(0, pad_h)
        padding = (left, top, pad_w - left, pad_h - top)

        return TF.pad(img_resized, padding, fill=bg_color, padding_mode='constant')


# ================= 模型定义 =================
class MetricViT(nn.Module):
    def __init__(self, num_classes, pretrained_name='google/vit-base-patch16-224'):
        super().__init__()
        # 使用 local_files_only 防止网络报错
        self.vit = ViTModel.from_pretrained(pretrained_name, local_files_only=True)
        self.hidden_dim = self.vit.config.hidden_size
        self.classifier = nn.Linear(self.hidden_dim, num_classes, bias=False)
        self.s = 30.0  # 缩放因子
        self.m = 0.35  # CosFace Margin (关键！迫使相似的卡分开)

    def forward(self, x, labels=None):
        outputs = self.vit(x)
        cls_token = outputs.last_hidden_state[:, 0]
        # L2 归一化，这对 Metric Learning 至关重要
        features = F.normalize(cls_token, p=2, dim=1)
        weight = F.normalize(self.classifier.weight, p=2, dim=1)

        # 计算余弦相似度 cos(theta)
        cosine = F.linear(features, weight)

        if self.training and labels is not None:
            # -------- CosFace 核心逻辑 --------
            # 把当前 batch 正确类别的余弦相似度减去 margin
            target_logits = cosine[torch.arange(x.size(0)), labels] - self.m
            # 把减去 margin 的值替换回去
            cosine[torch.arange(x.size(0)), labels] = target_logits

            logits = cosine * self.s
            return logits, features
        else:
            # 推理阶段直接返回
            logits = cosine * self.s
            return logits, features


# ================= 数据增强 =================
data_transforms = {
    'train': transforms.Compose([
        RandomBackgroundPad(p=0.4, scale_range=(0.85, 0.95)),
        PadToSquare(fill=(128, 128, 128)),
        transforms.Resize((224, 224)),
        # degrees: 微微旋转 | translate: 平移 | scale: 缩放 | shear: 错切(模拟轻微倾斜)
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=3),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
        # 模拟镜头失焦和画质变差 (高斯模糊和锐化)
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
        transforms.ColorJitter(brightness=(0.8, 1.3), contrast=(0.7, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# ================= 主函数 =================
def main():
    print(f"当前使用设备: {DEVICE}")
    print(f"可见显卡数量: {torch.cuda.device_count()} (应该显示 2)")

    if not os.path.exists(TRAIN_DIR):
        print(f"Error: 找不到训练目录 {TRAIN_DIR}")
        return

    # 1. 准备数据
    print("正在扫描数据集目录...")
    full_dataset = datasets.ImageFolder(TRAIN_DIR, transform=data_transforms['train'])
    num_classes = len(full_dataset.classes)
    print(f"检测到 {num_classes} 个类别")

    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)

    # 2. 初始化模型结构
    print("正在初始化模型...")
    # 请确保这个路径是你已经下载好的本地路径
    local_model_path = r"/home/martin/.cache/huggingface/hub/models--google--vit-base-patch16-224/snapshots/3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3"

    model = MetricViT(num_classes=num_classes, pretrained_name=local_model_path)

    # 冻结
    print("正在配置冻结层...")
    # 1. 先冻结所有参数
    for param in model.vit.parameters():
        param.requires_grad = False

    # 2. 精确解冻 Layer 10, 11
    for name, param in model.vit.named_parameters():
        # 解冻最后层 Encoder
        if any(f"encoder.layer.{i}" in name for i in range(4, 12)):
            param.requires_grad = True

        # 解冻 ViT 最后的归一化层 (HuggingFace 命名通常是 layernorm 或 pooler)
        # 注意：排除 encoder 内部的 layernorm，只解冻最外层的
        if name.startswith("layernorm") or name.startswith("pooler"):
            param.requires_grad = True

    # 3. 确保分类头一定是解冻的
    for param in model.classifier.parameters():
        param.requires_grad = True

    # 打印检查
    print("-" * 30)
    print("以下参数将被更新 (Check):")
    count_trainable = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"\t{name}")
            count_trainable += 1
    print(f"共 {count_trainable} 组参数参与训练。")
    print("-" * 30)

    # ========================================================
    # 加载旧权重 (Resume) - 必须在 DataParallel 之前
    # ========================================================
    if RESUME_PATH and os.path.exists(RESUME_PATH):
        print(f"正在加载恢复权重: {RESUME_PATH}")
        try:
            checkpoint = torch.load(RESUME_PATH, map_location="cpu")  # 先加载到 CPU
            new_state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=True)
            print("权重加载成功！")
        except Exception as e:
            print(f"加载失败: {e}，将从头开始训练。")

    # ========================================================
    # 移动到 GPU 并开启双卡并行
    # ========================================================
    model = model.to(DEVICE)
    if torch.cuda.device_count() > 1:
        print("开启 DataParallel 多卡并行加速...")
        model = nn.DataParallel(model)

    # ========================================================
    # 差异化学习率 (骨干小火，分类头大火)
    # ========================================================
    backbone_params = []
    head_params = []

    # 这里的 model 可能是 DataParallel，所以用 model.module 来访问内部成员
    # 如果没开启 DP，model 本身就是 MetricViT
    real_model = model.module if isinstance(model, nn.DataParallel) else model

    for name, param in real_model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    # 冻结训练
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},  # 骨干层微调 (很小)
        {'params': head_params, 'lr': 1e-4}  # 分类头学习 (标准)
    ], weight_decay=WEIGHT_DECAY)

    # 全训练
    # optimizer = optim.AdamW([
    #     {'params': backbone_params, 'lr': 3e-6},  # 骨干层微调 (很小)
    #     {'params': head_params, 'lr': 3e-5}  # 分类头学习 (标准)
    # ], weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    # 混合精度
    scaler = torch.amp.GradScaler('cuda')

    print("开始训练循环...")

    # 初始化最佳准确率
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}") as t:
            optimizer.zero_grad(set_to_none=True)

            for i, (inputs, labels) in enumerate(t):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                with torch.amp.autocast('cuda'):
                    logits, feats = model(inputs, labels)
                    loss = criterion(logits, labels) / GRAD_ACCUMULATION_STEPS

                scaler.scale(loss).backward()

                if (i + 1) % GRAD_ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                running_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(logits, dim=1)

                # 收集预测结果 (CPU)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                t.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(full_dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)

        # 获取当前的学习率进行打印
        current_lr_backbone = optimizer.param_groups[0]['lr']
        current_lr_head = optimizer.param_groups[1]['lr']

        print(
            f"Epoch {epoch + 1} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | LR(Back): {current_lr_backbone:.2e}")

        # 保存最佳模型 (保存 state_dict)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            # 注意：保存时要取 model.module (如果是 DP)
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, "best_model.pth")
            print(f"★ 保存最佳模型 (Acc: {best_acc:.4f})")

            best_backbone_dir = SAVE_MODEL_PATH + "_best"
            real_model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            real_model_to_save.vit.save_pretrained(best_backbone_dir)

        # 每个 epoch 结束后更新学习率
        scheduler.step()

    # 训练结束保存 backbone
    print(f"训练结束。保存最终 Backbone 到 {SAVE_MODEL_PATH} ...")
    real_model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    real_model_to_save.vit.save_pretrained(SAVE_MODEL_PATH)
    print("完成！")


if __name__ == '__main__':
    main()
