import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses

df = pd.read_excel(r"D:\Code\ML\Text\embedding\ebay_2023_data01.xlsx")
train_df = df.sample(frac=1, random_state=42)
# val_df = df.drop(train_df.index)

# 准备训练数据
train_data = [InputExample(texts=[row['ebay_text'], row['card_set'],
                                  row["program"], row["athlete"]]) for _, row in train_df.iterrows()]


for i, data in enumerate(train_data):
    print(i, data)
    if i == 10:
        break
