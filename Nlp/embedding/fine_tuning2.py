from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# 禁用 WandB (保持不变)
os.environ["WANDB_DISABLED"] = "true"

# 加载模型
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# 定义损失函数
train_loss = losses.MultipleNegativesRankingLoss(model)

# 加载数据集
df = pd.read_excel(r"D:\Code\ML\Text\embedding\ebay_2023_data01.xlsx")

# 划分训练集和验证集 (80% 训练, 20% 验证)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 准备训练数据
train_data = [
    InputExample(texts=[row['ebay_text'], row[field]])
    for _, row in train_df.iterrows()
    for field in ['card_set', 'program', 'athlete']
]

# 准备验证数据
val_data = [
    InputExample(texts=[row['ebay_text'], row[field]])
    for _, row in val_df.iterrows()
    for field in ['card_set', 'program', 'athlete']
]

# 创建数据加载器
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=64, num_workers=4)  # 调整 batch_size
val_dataloader = DataLoader(val_data, shuffle=False, batch_size=64)

# 定义评估器 (用于验证集评估)
evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(val_data, name='val')

# 训练模型
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=20,
          warmup_steps=100,
          optimizer_params={'lr': 2e-5},  # 调整学习率
          early_stopping_patience=3  # 连续3次评估无提升时停止
          )

# 保存模型
model.save(r"D:\Code\ML\Model\huggingface\all-mpnet-base-v2_fine_tag01")
