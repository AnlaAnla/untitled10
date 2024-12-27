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


def test(model, eval_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    with torch.no_grad():
        for batch in tqdm(eval_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            eval_loss += outputs.loss.item()
            logits = outputs.logits
            predicted = torch.argmax(logits, dim=1)

            # 用来瞎猜的代码
            # predicted = torch.ones_like(labels).to(device)
            eval_accuracy += (predicted == labels).sum().item() / len(labels)

        print(f'Eval Loss: {eval_loss / len(eval_loader):.4f}')
        print(f'Eval Accuracy: {eval_accuracy / len(eval_loader):.4f}')


def main():
    # 加载数据
    data = pd.read_csv(r"D:\Code\ML\Text\Classify\11metadata.csv")
    model_path = "sentence_judge_bert03"

    data = process_pd_data(data)
    data.reset_index(drop=True, inplace=True)  # 重置索引

    # 加载预训练tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)
    max_length = 512

    eval_dataset = TextClassificationDataset(data, tokenizer, max_length)
    eval_loader = DataLoader(eval_dataset, batch_size=32)

    # 加载预训练模型
    model = BertForSequenceClassification.from_pretrained(model_path,
                                                          num_labels=len(data['label'].unique()))

    test(model, eval_loader)


if __name__ == '__main__':
    main()
