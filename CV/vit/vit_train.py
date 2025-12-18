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
        if ("encoder.layer.9" in name or "encoder.layer.10" in name
                or "encoder.layer.11" in name):
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

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},  # 骨干层微调 (很小)
        {'params': head_params, 'lr': 1e-4}  # 分类头学习 (标准)
    ], weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 混合精度
    scaler = torch.amp.GradScaler('cuda')

    print("开始训练循环...")

    # 初始化最佳准确率
    best_acc = 0.85

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}") as t:
            for i, (inputs, labels) in enumerate(t):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    logits, _ = model(inputs)
                    loss = criterion(logits, labels)

                # 缩放损失
                scaler.scale(loss / GRAD_ACCUMULATION_STEPS).backward()

                if (i + 1) % GRAD_ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

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

        # 每个 epoch 结束后更新学习率
        scheduler.step()

    # 训练结束保存 backbone
    print(f"训练结束。保存最终 Backbone 到 {SAVE_MODEL_PATH} ...")
    real_model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    real_model_to_save.vit.save_pretrained(SAVE_MODEL_PATH)
    print("完成！")


if __name__ == '__main__':
    main()
