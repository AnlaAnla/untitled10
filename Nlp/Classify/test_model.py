import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd


def get_embedding(text, model, tokenizer, device):
    # 对文本进行tokenize
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # 使用模型进行预测
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # 获取最后一层所有token的隐藏状态
        hidden_states = outputs.hidden_states[-1]

        # 方法一：获取 [CLS] token 的嵌入向量 (索引为 0)
        cls_embedding = hidden_states[:, 0, :].cpu().numpy()

        # 方法二：获取所有token的平均嵌入向量
        # mean_embedding = torch.mean(hidden_states, dim=1).cpu().numpy()

    return cls_embedding


def predict(text, model, tokenizer, device):
    # 对标签进行解码
    label_mapping = {idx: label for label, idx in model.config.label2id.items()}

    # 对文本进行tokenize
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
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
        print(logits)
        predicted_label = torch.argmax(logits, dim=1).item()

    # 解码标签
    predicted_label_text = label_mapping[predicted_label]

    return predicted_label_text


def main():
    model_path = "sentence_judge_bert03"
    # 加载预训练tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)
    max_length = 512

    # 加载训练好的模型
    model = BertForSequenceClassification.from_pretrained(model_path)
    # model = BertForSequenceClassification.from_pretrained('sentence_judge_bert01')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 测试样例文本
    texts = [
        "啊来点好卡吧快",
        "还有下一句开完一起吗",
        "但没有见过带编就是那种99199的什么",
        "谢斯克的150编",
        "我真的笑了",

             ]

    for text in texts:
        # 进行预测
        # predicted_label = predict(text, model, tokenizer, device)
        # print(f"{text} :  {predicted_label}")

        embedding = get_embedding(text, model, tokenizer, device)
        print(f"Embedding shape: {embedding.shape}")
        print('+'*30)


if __name__ == '__main__':
    main()
