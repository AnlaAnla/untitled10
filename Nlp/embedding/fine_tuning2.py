from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
import os

# 禁用 WandB (如果你不想使用它)
os.environ["WANDB_DISABLED"] = "true"

# 加载模型
model = SentenceTransformer(r"D:\Code\ML\Model\huggingface\all-MiniLM-L6-v2_fine_tag6")

# 定义损失函数
train_loss = losses.MultipleNegativesRankingLoss(model)

# 加载数据集
df = pd.read_excel(r"D:\Code\ML\Text\embedding\ebay_2023_data01.xlsx")
train_df = df.sample(frac=0.9, random_state=42)  # 80% 训练集
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



# 创建数据加载器
batch_size = 128  # 根据你的 GPU 内存调整
accumulation_steps = 2  # 梯度累积
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

# 创建评估器
evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
    val_data, name='val-set'
)

# 训练设置
num_epochs = 30
num_training_steps = num_epochs * len(train_dataloader) // accumulation_steps  # 考虑梯度累积
optimizer = AdamW(model.parameters(), lr=3e-4)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps  # 10% 的预热步数
)

# 训练模型 (修改部分)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],  # 关键修改：只传入 dataloader 和 loss
    epochs=num_epochs,
    evaluator=evaluator,
    evaluation_steps=500,
    output_path="path/to/save/best_model",
    save_best_model=True,
    optimizer_class=AdamW,  # 添加 optimizer_class
    optimizer_params={'lr': 2e-4},  # 添加 optimizer_params
    scheduler='WarmupLinear',  # warmuplinear 变成了 WarmupLinear (首字母大写)
    warmup_steps=int(0.1 * num_training_steps),  # 10% 的预热步数, 明确指定warmup_steps
    weight_decay=0.01,  # 添加weight_decay，AdamW的默认值
    use_amp=True,  # 添加自动混合精度，加快训练速度(如果你的GPU支持)
)

# 保存最终模型
model.save(r"D:\Code\ML\Model\huggingface\all-MiniLM-L6-v2_fine_tag7")
