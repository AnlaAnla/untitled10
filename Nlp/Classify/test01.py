import pandas as pd
import numpy as np
import os

data = pd.read_csv(r"D:\Code\ML\Text\Classify\11metadata.csv", encoding="utf-8")
data = data[:1035]

# 假设您已经有了 data 这个 DataFrame
# data['label'] 是您需要处理的那一列

# 首先创建一个布尔掩码,用于识别需要替换为 0 的值
mask_zero = data['label'].isna() | (data['label'].str.strip() == '')

# 创建另一个布尔掩码,用于识别需要替换为 1 的值
mask_one = ~mask_zero & (data['label'].str.strip() != '')

# 使用 numpy 的 where 函数进行值替换
data['label'] = data['label'].where(mask_one, other=np.nan)
data['label'] = data['label'].where(mask_zero, other=1)
data['label'] = data['label'].fillna(0).astype(int)

print(data)
