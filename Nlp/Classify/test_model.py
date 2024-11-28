import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd


def predict(text, model, tokenizer, max_length, device):
    # 对标签进行解码
    label_mapping = {idx: label for label, idx in model.config.label2id.items()}

    # 对文本进行tokenize
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # 使用模型进行预测
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()

    # 解码标签
    predicted_label_text = label_mapping[predicted_label]

    return predicted_label_text


def main():
    # 加载预训练tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    max_length = 512

    # 加载训练好的模型
    model = BertForSequenceClassification.from_pretrained('bert_result01')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 测试样例文本
    texts = ["演员演技超赞", '整体性能也不错', "环境很吵闹", '体验不佳', "公司的管理制度", '虽然这本书的内容很有趣,但是写作风格有些枯燥乏味。']

    for text in texts:
        # 进行预测
        predicted_label = predict(text, model, tokenizer, max_length, device)
        print(f"{text} :  {predicted_label}")


if __name__ == '__main__':
    main()
