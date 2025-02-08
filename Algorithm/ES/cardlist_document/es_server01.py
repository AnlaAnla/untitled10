from elasticsearch import Elasticsearch
import re
import pandas as pd

# 连接到 ES (确保配置正确)
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# 从 CSV 文件中动态获取 program_new 和 card_set 的关键词列表 (确保 csv_file 路径正确)
csv_file = r"D:\Code\ML\Text\test\paniniamerica_checklist_refresh.csv"
df_keywords = pd.read_csv(csv_file)
program_keywords = df_keywords['program_new'].dropna().unique().tolist()
card_set_keywords = df_keywords['card_set'].dropna().unique().tolist()


def search_card_by_ebay_text(ebay_text):
    """
    使用 eBay 文本搜索卡牌，返回最匹配的前 10 个 bgs_title 字段。
    """
    extracted_details = extract_card_details_regex_from_ebay_text(ebay_text)
    athlete_name = extract_athlete_name_from_ebay_text(ebay_text)

    es_query = {
        "query": {
            "bool": {
                "must": [],
                "should": [],
                "minimum_should_match": 1
            }
        },
        "size": 15  # 限制返回结果数量为 10
    }

    if extracted_details.get('year'):
        es_query["query"]["bool"]["must"].append({"term": {"year": extracted_details['year']}})
    if extracted_details.get('card_number'):
        es_query["query"]["bool"]["must"].append({"term": {"card_number": extracted_details['card_number']}})

        # 在 SHOULD 子句中使用 match 查询 (改回 match 查询，更灵活)
    if card_set_keywords:
        es_query["query"]["bool"]["should"].append(
            {"match": {"card_set": {"query": ebay_text, "boost": 5}}})  # 使用 match 查询，高权重
    if program_keywords:
        es_query["query"]["bool"]["should"].append(
            {"match": {"program_new": {"query": ebay_text, "boost": 4}}})  # 使用 match 查询，较高权重

        # Athlete 仍然是 match 查询，放在 SHOULD 子句
    if athlete_name:
        es_query["query"]["bool"]["should"].append(
            {"match": {"athlete": {"query": athlete_name, "fuzziness": "AUTO", "boost": 1.5}}})

    if not es_query["query"]["bool"]["must"] and not es_query["query"]["bool"]["should"]:
        return []

    try:
        res = es.search(index="cards", body=es_query)
        hits = res['hits']['hits']
        if hits:
            top_bgs_titles = [hit['_source'].get('bgs_title') for hit in hits]  # 提取前 10 个结果的 bgs_title
            return top_bgs_titles  # 返回 bgs_title 列表
        else:
            return []  # 没有找到匹配结果，返回空列表
    except Exception as e:
        print(f"ES query error: {e}")
        return []  # 查询出错返回空列表


def extract_card_details_regex_from_ebay_text(ebay_text):
    """
    使用正则表达式从 eBay 文本中提取年份和卡号.
    """
    year_regex = r"\b(20\d{2})(?:-\d{2})?\b"
    card_number_regex = r"(?:\s|^)#(\d+)"

    extracted_details = {}

    year_match = re.search(year_regex, ebay_text, re.IGNORECASE)
    if year_match:
        extracted_details['year'] = int(year_match.group(1))

    card_number_match = re.search(card_number_regex, ebay_text)
    if card_number_match:
        extracted_details['card_number'] = card_number_match.group(1)

    return extracted_details


def extract_athlete_name_from_ebay_text(ebay_text):
    """
    更精确的运动员姓名提取.
    """
    rc_suffix_regex = r"(?i)\s*(?:Rookie Card|\(RC\))$"
    text_without_suffix = re.sub(rc_suffix_regex, "", ebay_text).strip()

    text_without_year_number = re.sub(r"\b(20\d{2})(?:-\d{2})?\b", "", text_without_suffix, flags=re.IGNORECASE)
    text_without_year_number_card = re.sub(r"#(\d+)\b", "", text_without_year_number)

    brand_keywords_regex = r"\b(Donruss|Obsidian|Panini|Prime|Contenders)\b"
    text_without_brand_year_number_card = re.sub(brand_keywords_regex, "", text_without_year_number_card,
                                                 flags=re.IGNORECASE)

    athlete_name = text_without_brand_year_number_card.strip()
    return athlete_name.strip()


# 示例用法
example_list = [
    "2023 knicks jalen brunson panini prizm #9",
    "2023 kings harrison barnes prnini prizm #221"

]


for ebay_text_example in example_list:
    # ebay_text_example = "2023 Panini Prizm Dameon Pierce Pandora /400 Lot7"
    top_10_titles = search_card_by_ebay_text(ebay_text_example)
    if top_10_titles:
        print(f"text: '{ebay_text_example}':")
        for title in top_10_titles:
            print(f"- {title}")
    else:
        print(f"No matching cards found for eBay text: '{ebay_text_example}'")

    print("_"*22)
