import re

def preprocess_year_num(text):
    # 尝试匹配 "20xx-yy"、"20xx" 或 "xx-yy" 格式的年份
    year_match = re.search(r'\b(20\d{2})-(\d{2})\b|\b(20\d{2})\b|\b(\d{2})-(\d{2})\b', text)
    num_match = re.search(r'#([A-Za-z0-9/-]+)', text)  # 修改的正则

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

    return {
        'year': year,
        'card_num': card_num
    }


# 测试用例
print(preprocess_year_num("2023-24 #123"))
print(preprocess_year_num("2024 #456"))
print(preprocess_year_num("23-24 #ABC"))
print(preprocess_year_num("#046/100"))
print(preprocess_year_num("#1-7"))
print(preprocess_year_num("#XYZ-abc"))
print(preprocess_year_num("#12-34-56"))
print(preprocess_year_num("2021 #A1-B2"))
print(preprocess_year_num("#AbC-123"))