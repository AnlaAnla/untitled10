import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses

df = pd.read_excel(r"D:\Code\ML\Text\embedding\ebay_2023_data01.xlsx")
train_df = df.sample(frac=0.8, random_state=42)  # 80% 训练集
val_df = df.drop(train_df.index)  # 剩余 20% 作为验证集

# 准备训练数据和验证数据
train_data = [
    InputExample(texts=[row['ebay_text'], row[field]])
    for _, row in train_df.iterrows()
    for field in ['card_set', 'program', 'athlete']
]
val_data = [
    InputExample(texts=[row['ebay_text'], row[field]], label=1.0)  # 添加 label=1.0
    for _, row in val_df.iterrows()
    for field in ['card_set', 'program', 'athlete']
]


for example in val_data[:10]:
    print(example.texts, example.label)