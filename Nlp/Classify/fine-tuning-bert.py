import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm


# 定义数据集类
class TextClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.loc[index, 'sentence']
        label = self.data.loc[index, 'label']

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def process_pd_data(data):
    data = data.drop(['file_name'], axis=1)
    # data = data[:6000]

    # 首先创建一个布尔掩码,用于识别需要替换为 0 的值
    mask_zero = data['label'].isna() | (data['label'].str.strip() == '')

    # 创建另一个布尔掩码,用于识别需要替换为 1 的值
    mask_one = ~mask_zero & (data['label'].str.strip() != '')

    # 使用 numpy 的 where 函数进行值替换
    data['label'] = data['label'].where(mask_one, other=np.nan)
    data['label'] = data['label'].where(mask_zero, other=1)
    data['label'] = data['label'].fillna(0).astype(int)

    return data


def train(model, epochs,
          save_path, train_loader, eval_loader,
          lr_rate=5e-5,):
    # 冻结BERT模型的部分层
    # 冻结embeddings层
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False

    # 冻结编码器层
    for layer in range(6):
        for param in model.bert.encoder.layer[layer].parameters():
            param.requires_grad = False

    # 设置优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate)
    # loss_fn = torch.nn.CrossEntropyLoss()

    # 训练模型
    epochs = epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        eval_loss, eval_accuracy = 0, 0
        model.eval()
        with torch.no_grad():
            for batch in tqdm(eval_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                eval_loss += outputs.loss.item()
                logits = outputs.logits
                predicted = torch.argmax(logits, dim=1)
                eval_accuracy += (predicted == labels).sum().item() / len(labels)

        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Train Loss: {train_loss / len(train_loader):.4f}')
        print(f'Eval Loss: {eval_loss / len(eval_loader):.4f}')
        print(f'Eval Accuracy: {eval_accuracy / len(eval_loader):.4f}')

    # 保存模型
    model.save_pretrained(save_path)



def main():
    # 加载数据
    data = pd.read_csv(r"D:\Code\ML\Text\Classify\11metadata.csv")
    data = process_pd_data(data)

    # 对标签进行编码
    # label_mapping = {label: idx for idx, label in enumerate(data['label'].unique())}
    # data['label'] = data['label'].map(label_mapping)

    # 划分数据集
    train_data, eval_data = train_test_split(data, test_size=0.1, random_state=42)
    train_data.reset_index(drop=True, inplace=True)  # 重置索引
    eval_data.reset_index(drop=True, inplace=True)  # 重置索引

    # 加载预训练tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    max_length = 512

    # 创建数据集
    train_dataset = TextClassificationDataset(train_data, tokenizer, max_length)
    eval_dataset = TextClassificationDataset(eval_data, tokenizer, max_length)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=15, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=15)

    # 加载预训练模型
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=len(data['label'].unique()))

    save_path = "sentence_judge_bert03"
    # 开始训练
    train(model, epochs=4, lr_rate=5e-5,
          save_path=save_path,
          train_loader=train_loader,
          eval_loader=eval_loader
          )
    tokenizer.save_pretrained(save_path)


if __name__ == '__main__':
    main()
