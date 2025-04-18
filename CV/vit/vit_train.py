import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import ImageFile
from transformers import ViTForImageClassification, ViTConfig
import os
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


# 检查 CUDA 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# 数据集路径
data_dir = r"D:\Code\ML\Embedding\img_vec\checklist2023_classes\train"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "train")

# 超参数
image_size = 224
batch_size = 64  # 根据您的显存调整
learning_rate = 2e-4
epochs = 10
weight_decay = 0.01
num_classes = len(os.listdir(train_dir))  # 自动检测类别数量
gradient_accumulation_steps = 2  # 模拟更大的 batch size
feature_dim = 768  # ViT-Base 的特征维度

# print(f"Number of classes: {num_classes}")h

# 数据增强
data_transforms = {
    'train': transforms.Compose([
        # transforms.Resize((image_size, image_size)),
        transforms.RandAugment(),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        # transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 加载数据集
image_datasets = {
    'train': datasets.ImageFolder(train_dir, data_transforms['train']),
    'val': datasets.ImageFolder(val_dir, data_transforms['val'])
}
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
    'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}


# 训练循环
def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_labels = []
            all_preds = []

            with tqdm(dataloaders[phase], desc=f'{phase} Epoch {epoch}/{num_epochs - 1}', unit='batch') as tepoch:
                for i, (inputs, labels) in enumerate(tepoch):
                    inputs = inputs.to(device, dtype=torch.float32)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)  # 现在 outputs 是 ImageClassifierOutput
                        logits = outputs.logits  # 从 outputs 中提取 logits
                        loss = criterion(logits, labels)  # 使用 logits 计算损失

                    if phase == 'train':
                        scaler.scale(loss / gradient_accumulation_steps).backward()

                        if (i + 1) % gradient_accumulation_steps == 0:
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()

                    _, preds = torch.max(logits, 1)  # 使用 logits 进行预测

                    running_loss += loss.item() * inputs.size(0)

                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())

                    tepoch.set_postfix(loss=running_loss / ((i + 1) * dataloaders[phase].batch_size))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = accuracy_score(all_labels, all_preds)
            epoch_f1 = f1_score(all_labels, all_preds, average='macro')

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                torch.save(best_model_wts, 'best_model.pth')

            if phase == 'train':
                scheduler.step()

    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(torch.load('best_model.pth'))
    return model


if __name__ == '__main__':
    # 加载 ViT 模型
    config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
    config.num_labels = num_classes  # 设置分类数量
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224',
                                                      config=config,
                                                      ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))


    # 冻结 ViT 模型的部分层
    for name, param in model.named_parameters():
        if 'classifier' in name or 'layernorm' in name or 'encoder.layer.11' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    # model.classifier.weight.requires_grad = True
    # model.classifier.bias.requires_grad = True

    model = model.to(device)

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # 学习率衰减

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 混合精度训练
    scaler = torch.amp.GradScaler('cuda')
    # scaler = torch.cuda.amp.GradScaler()

    # 训练模型
    model_ft = train_model(model, criterion, optimizer, scheduler, num_epochs=epochs)

    print("Finished Training")
    print("==============")
    print(model_ft)

# 特征提取函数
# def extract_features(model, dataloader):
#     model.eval()  # 设置为评估模式
#     features = []
#     labels = []
#     with torch.no_grad():
#         for inputs, lbs in tqdm(dataloader, desc="Extracting Features"):
#             inputs = inputs.to(device)
#             outputs = model(inputs).pooler_output  # 或者使用 .last_hidden_state 获取所有 token 的特征
#             features.append(outputs.cpu().numpy())
#             labels.append(lbs.numpy())  # 保存标签
#     return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)  # 返回特征和标签
#
#
# # 创建用于特征提取的 DataLoader，不进行 shuffle
# feature_dataloader = DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4,
#                                 pin_memory=True)
#
# # 提取特征
# features, labels = extract_features(model_ft, feature_dataloader)
#
# print(f"Extracted features shape: {features.shape}")
# print(f"Extracted labels shape: {labels.shape}")
#
# # 保存特征
# np.save('features.npy', features)
# np.save('labels.npy', labels)
#
# print("Features saved to features.npy")
# print("Labels saved to labels.npy")