import pandas as pd
import numpy as np
import os
import re


def process_pd_data(data_path):
    # 首先创建一个布尔掩码,用于识别需要替换为 0 的值
    data = pd.read_csv(data_path)
    mask_zero = data['label'].isna() | (data['label'].str.strip() == '')
    data = data.drop(data[mask_zero].index)
    data.reset_index(drop=True, inplace=True)  # 重置索引

    return data


csv_path = r"D:\Code\ML\Text\Classify\judge_data\judge_metadata.csv"
data = process_pd_data(csv_path)

base_data_dir = r"D:\Code\ML\Text\Classify\judge_data\audio"
for i in range(len(data)):
    audio_path = os.path.join(base_data_dir, data['file_name'][i])
    data_text = data['sentence'][i]
    data_key = data['label'][i]


text = "ad afagg, agrg ,asd"

# 使用正则表达式分割字符串
pattern = r'[, ]+'
parts = re.split(pattern, text)

# 去除空字符串和前后空格
cleaned_parts = [part.strip() for part in parts if part.strip()]

print(cleaned_parts)