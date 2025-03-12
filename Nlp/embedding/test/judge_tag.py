import numpy as np
import pandas as pd


def get_data_list(data_path):
    data_list = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            tag = line.strip()
            if tag:  # 排除空行
                data_list.append(tag)
    print('length: ', len(data_list))
    return data_list


if __name__ == '__main__':
    match_ebay_data = pd.read_excel(r"D:\Code\ML\Text\embedding\ebay_2023_data01_test.xlsx")

    tag_list = ['program', "card_set", "athlete"]
    for tag in tag_list:
        match_tag = list(match_ebay_data[tag].dropna().unique())
        card_set_database = get_data_list(f"D:\Code\ML\Text\checklist_tags\\2023\\{tag}.txt")

        print('------------------ ', tag)
        for tag in match_tag:
            if tag not in card_set_database:
                print(tag)

    print('end')

    # =====================================

    # tag_list = ['program', "card_set", "athlete"]
    # new_match_ebay_data = match_ebay_data.copy()
    #
    # # 创建一个布尔掩码，初始值为 True（所有行都保留）
    # mask = pd.Series([True] * len(match_ebay_data), index=match_ebay_data.index)
    #
    # for tag in tag_list:
    #     for i, data in match_ebay_data.iterrows():
    #         word_in_text_num = 0
    #         ebay_text = data['ebay_text'].lower()
    #         #  处理 NaN 值
    #         tag_value = data[tag]
    #         if pd.isna(tag_value):
    #             card_set_words = []  # 或者根据你的逻辑进行其他处理
    #         else:
    #             card_set_words = str(tag_value).lower().split()  # 确保转换为字符串
    #
    #         for word in card_set_words:
    #             if word in ebay_text:
    #                 word_in_text_num += 1
    #
    #         # 如果card set里面的单词一个都没有在ebay text 里面
    #         if word_in_text_num == 0:
    #             print(data['ebay_text'], ' |---| ', data[tag])
    #             mask[i] = False  # 将掩码对应位置设置为 False，表示要删除该行
    #
    # # 使用布尔掩码过滤 DataFrame
    # new_match_ebay_data = new_match_ebay_data[mask]
    #
    # print('length: ', len(match_ebay_data))
    # print('new length: ', len(new_match_ebay_data))
    #
    # if len(new_match_ebay_data) < len(match_ebay_data):
    #     new_match_ebay_data.to_excel(r"D:\Code\ML\Text\embedding\ebay_2023_data01_refresh.xlsx", index=False)
    # print('end')
