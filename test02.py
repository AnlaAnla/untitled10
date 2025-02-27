import pandas as pd
import numpy as np
from thefuzz import fuzz, process


def filter_dataframe_optimized(dataframe, filter_dict):
    """
    根据给定的字典中的字段筛选 DataFrame

    Args:
        dataframe: 要筛选的 Pandas DataFrame。
        filter_dict: 包含筛选条件的字典。键是 DataFrame 的列名，值是筛选条件。
                     支持的键: 'program_new', 'card_num', 'athlete_new'
                     如果值为 None 或空字符串，则忽略该筛选条件。
                     如果键是 'card_num' 且值不是纯数字字符串，则忽略。

    Returns:
        筛选后的 DataFrame。
    """

    mask = True  # 初始 mask 为 True

    for column, value in filter_dict.items():
        if not value:  # 等价于 if value is None or value == "":
            continue

        if column == 'card_num':
            if not isinstance(value, str) or not value.isdigit():
                continue
            value = str(value)  # card_num 转为字符串
        elif column not in ('program_new', 'athlete_new'):  # 优化点1
            continue

        if column in dataframe.columns:  # 优化点2
            mask = mask & (dataframe[column] == value)

    return dataframe[mask]


checklist_2023 = pd.read_csv(r"D:\Code\ML\Text\card\checklist_2023.csv")
output = {'program_new': 'Prizm', 'card_num': '28', 'athlete_new': 'Stephen Curry'}

filtered_name_data = list(filter_dataframe_optimized(checklist_2023, output)['card_set'])
print(filtered_name_data)

matches = process.extract("2023-24 Prizm Monopoly Stephen Curry SILVER Prizm Card #28 Warriors Star!", filtered_name_data, scorer=fuzz.partial_ratio, limit=20)
print(matches)

print()