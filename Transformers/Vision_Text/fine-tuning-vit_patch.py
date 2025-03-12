import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel, AdamW, get_linear_schedule_with_warmup
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 超参数
BATCH_SIZE = 32  # 根据你的 GPU 内存调整
EPOCHS = 10
LEARNING_RATE = 2e-5
WARMUP_STEPS = 100
EARLY_STOPPING_PATIENCE = 3

# 加载 CLIP 模型和处理器
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# 数据集类
class EbayCardDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"])
        # 将多个文本字段合并, 也可以分别处理各个字段
        text = f"{item['ebay_text']} Series: {item['card_set']}, Program: {item['program']}, Athlete: {item['athlete']}"

        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}  # 去掉批次维度
        return inputs


# 加载数据
df = pd.read_excel("ebay_2023_data01.xlsx")
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 将数据转换为列表
train_data = train_df.to_dict("records")
val_data = val_df.to_dict("records")

train_dataset = EbayCardDataset(train_data, processor)
val_dataset = EbayCardDataset(val_data, processor)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)

# 优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

# 早停
best_val_loss = float('inf')
epochs_no_improve = 0

# 训练循环
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

        # 计算 InfoNCE 损失
        labels = torch.arange(logits_per_image.size(0)).to(device)
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2

        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Average training loss: {avg_train_loss:.4f}")

    # 验证
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text

            labels = torch.arange(logits_per_image.size(0)).to(device)
            loss_i = F.cross_entropy(logits_per_image, labels)
            loss_t = F.cross_entropy(logits_per_text, labels)
            loss = (loss_i + loss_t) / 2
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)
    print(f"Average validation loss: {avg_val_loss:.4f}")

    # 早停检查
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        # 保存最佳模型
        torch.save(model.state_dict(), "best_model.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve == EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered")
            break

# 加载最佳模型
# model.load_state_dict(torch.load("best_model.pth"))
# model.eval()  # 切换到推理模式

# 构建向量库 (示例，使用 FAISS)
# import faiss
# import numpy as np
#
# # 假设你有一个 series_cardset_data 列表，包含所有系列和卡种的文本
# series_cardset_data = [
#     {"text": "Bowman Chrome Baseball"},
#     {"text": "Topps Series 1 Baseball"},
#     # ... 更多系列和卡种
# ]
#
# # 将文本编码为向量
# text_embeddings = []
# with torch.no_grad():
#     for item in series_cardset_data:
#         inputs = processor(text=[item["text"]], return_tensors="pt", padding=True).to(device)
#         text_embed = model.get_text_features(**inputs)
#         text_embeddings.append(text_embed.cpu().numpy())
#
# text_embeddings = np.concatenate(text_embeddings, axis=0)
#
# # 创建 FAISS 索引 (这里使用最简单的 IndexFlatL2)
# dimension = text_embeddings.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(text_embeddings)


# 搜索 (示例)
# def search(image_path, query_text, top_k=5):
#     image = Image.open(image_path)
#     inputs = processor(text=[query_text], images=image, return_tensors="pt", padding=True).to(device)
#     with torch.no_grad():
#         image_embed = model.get_image_features(**inputs)  # 图像向量
#         text_embed = model.get_text_features(**{k: v for k, v in inputs.items() if k != 'pixel_values'})  # 文本向量
#     # 也可以将图像和文本向量合并成一个向量， 再搜索
#     search_vector = image_embed.cpu().numpy()  # or text_embed.cpu().numpy() or concatenate them
#
#     D, I = index.search(search_vector, top_k)  # D: 距离, I: 索引
#     results = [series_cardset_data[i] for i in I[0]]
#     return results
#
#
# # 使用示例
# image_path = "path/to/your/test_image.jpg"
# query_text = "A 2023 Bowman Chrome card of a baseball player"
# results = search(image_path, query_text)
# print(results)