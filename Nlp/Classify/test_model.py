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

        "25杠25编,联盟logo,小虎,",
             "EDG,1115编,现场",
             "队标无编,刚才还有15编,这来个无编,凑套呢,",
             "就一个卡签呢",
             "还是个垃圾人的",
             "来,球球,NIP谁啊 卓哥,哎呦,我去,这个10编的",
             "03,10编,卓哥,大窗",
             "现场无编原神,Base,",
             "来 LPL结束,",
             "出了尺帝的签字,",
             "尺帝的15编签字,来看LCK,",
             "卓哥那个Patch切得还挺好呢,",
             "来,Closer,Keria现场",
             "2025编,Keria",
             "1-1,",
             "新打野,05,15编,现场,",
             "别着急啊 还有一盒呢",
             "呃,原神双签可接受,",
             "兑换卡,来,Cars,",
             "21,25编,MVP特卡,"

             ]

    for text in texts:
        # 进行预测
        predicted_label = predict(text, model, tokenizer, max_length, device)
        print(f"{text} :  {predicted_label}")
        print('+'*30)


if __name__ == '__main__':
    main()
