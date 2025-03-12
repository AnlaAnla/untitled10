import re
import pandas as pd

def preprocess_year_num(text:str):
    # 尝试匹配 "20xx-yy"、"20xx" 或 "xx-yy" 格式的年份
    year_match = re.search(r'\b(20\d{2})-(\d{2})\b|\b(20\d{2})\b|\b(\d{2})-(\d{2})\b', text)

    year = ""
    if year_match:
        if year_match.group(1):
            year = year_match.group(1)
        elif year_match.group(3):
            year = year_match.group(3)
        elif year_match.group(4):
            year = "20" + year_match.group(4)
            if int(year) > 2099 or int(year) < 2000:
                year = ""

    card_num = ""
    word_list = text.lower().split()
    if 'base' in word_list or 'rookie' in word_list or 'rc' in word_list:
        num_match = re.search(r'#([A-Za-z0-9/-]+)', text)  # 修改的正则
        if num_match:
            card_num_part = num_match.group(1)
            if "/" in card_num_part:  # 首先检查是否有斜杠
                card_num = ""
            elif "-" in card_num_part:
                parts = card_num_part.split("-")
                if all(part.isdigit() for part in parts):
                    card_num = parts[0]
                else:
                    card_num = card_num_part
            else:
                card_num = card_num_part
    else:
        card_num = ""

    return {
        'year': year,
        'card_num': card_num
    }

test_data = pd.read_excel("D:\Code\ML\Text\embedding\ebay_2023_data01_test3.xlsx")
length = len(test_data)

yes_year = 0
yes_num = 0

not_card_num = 0

for row in test_data.iterrows():

    year_num = preprocess_year_num(row[1]['ebay_text'])

    if year_num['year'] == str(row[1]['year']):
        yes_year += 1

    if year_num['card_num'] == '' or not year_num['card_num'].isdigit():
        not_card_num += 1
    elif year_num['card_num'] == str(row[1]['card_number']):
        yes_num += 1
    else:
        print(f"{row[1]['ebay_text']} [{row[1]['card_number']}]")
        print(year_num['card_num'])
        print()

print(f" {yes_year/length} [{yes_year}/{length}]")

print(f"yes num: {yes_num}")
print(f" not num: {not_card_num}")
print(f" {yes_num/length} [{yes_num}/{length}]")
print(f" {yes_num/(length-not_card_num)} [{yes_num}/{length - not_card_num}]")

