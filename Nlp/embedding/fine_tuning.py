from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd
import os


# 禁用 WandB (如果你不想使用它)
os.environ["WANDB_DISABLED"] = "true"

# 加载模型
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model = SentenceTransformer(r"D:\Code\ML\Model\huggingface\all-MiniLM-L6-v2_fine_cardSet3")

# 定义损失函数
train_loss = losses.MultipleNegativesRankingLoss(model)

# 加载数据集
df = pd.read_excel(r"D:\Code\ML\Text\embedding\ebay_2023_data01.xlsx")
train_df = df.sample(frac=1, random_state=42)
# val_df = df.drop(train_df.index)

# 准备训练数据
train_data = [
    InputExample(texts=[row['ebay_text'], row[field]])
    for _, row in train_df.iterrows()
    for field in ['card_set', 'program', 'athlete']
]
# val_data = [InputExample(texts=[row['ebay_text'], row['card_set']]) for _, row in val_df.iterrows()]

# 创建数据加载器
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=256)
# val_dataloader = DataLoader(val_data, shuffle=False, batch_size=32)

# 训练模型
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=25,
          warmup_steps=100,
          optimizer_params={'lr': 2e-5},
          )

# 保存模型
model.save(r"D:\Code\ML\Model\huggingface\all-MiniLM-L6-v2_fine_cardSet4")
