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

    # name_path = r"D:\Code\ML\Text\checklist_tags\2023\athlete.txt"
    # card_set_database = set(get_data_list(name_path))
    #
    match_ebay_data = pd.read_excel(r"D:\Code\ML\Text\embedding\ebay_2023_data01.xlsx")
    # match_tag = list(match_ebay_data['athlete'].dropna().unique())
    #
    # for tag in match_tag:
    #     if tag not in card_set_database:
    #         print(tag)
    #
    # print('end')
    # 创建一个示例 DataFrame

    # =====================================

    tag_list = ['program', "card_set", "athlete"]
    new_match_ebay_data = match_ebay_data.copy()

    for tag in tag_list:
        for i, data in enumerate(match_ebay_data.iterrows()):
            word_in_text_num = 0
            ebay_text = data[1]['ebay_text'].lower()
            card_set_words = data[1][tag].lower().split()
            for word in card_set_words:
                if word in ebay_text:
                    word_in_text_num += 1

            # 如果card set里面的单词一个都没有在ebay text 里面
            if word_in_text_num == 0:
                print(data[1]['ebay_text'], ' |---| ', data[1][tag])
                new_match_ebay_data = new_match_ebay_data.drop(i)

    print('length: ', len(match_ebay_data))
    print('new length: ', len(new_match_ebay_data))

    new_match_ebay_data.to_excel(r"D:\Code\ML\Text\embedding\ebay_2023_data01_test_refresh.xlsx", index=False)
    print('end')
