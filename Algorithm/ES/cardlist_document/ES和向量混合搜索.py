from elasticsearch import Elasticsearch
import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util

bert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# 连接到 ES (确保配置正确)
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# 从 CSV 文件中动态获取 program_new 和 card_set 的关键词列表 (确保 csv_file 路径正确)
csv_file = r"D:\Code\ML\Text\test\paniniamerica_checklist_refresh_norepeat.csv"
df_keywords = pd.read_csv(csv_file)
program_keywords = df_keywords['program_new'].dropna().unique().tolist()
card_set_keywords = df_keywords['card_set'].dropna().unique().tolist()


def search_card_by_ebay_text(ebay_text):
    """
    混合搜索：ES 初步召回 + BERT 精细排序
    """
    # 1. ES 初步召回 (Top N 候选卡片)
    top_n_candidates = get_top_n_es_candidates(ebay_text, n=100)  # 假设 get_top_n_es_candidates 函数返回 Top N 候选卡片

    if not top_n_candidates:
        return []  # ES 没有召回结果，直接返回空列表

    # 2. BERT 向量相似度计算和精细排序
    ebay_text_embedding = bert_model.encode(ebay_text, convert_to_tensor=True)  # 生成 eBay 文本的向量
    candidate_scores = []
    for candidate_card in top_n_candidates:
        bgs_title = candidate_card['_source'].get('bgs_title', '')
        if bgs_title is None:  # 检查 bgs_title 是否为 None
            similarity_score = -1  # 如果 bgs_title 为 None，赋予较低的相似度得分 -1，并跳过 BERT 编码
            print(f"Warning: bgs_title is None for card ID: {candidate_card['_id']}. Skipping BERT encoding.")  # 打印警告信息
        else:  # bgs_title 不为 None，进行 BERT 编码和相似度计算
            bgs_title_embedding = bert_model.encode(bgs_title, convert_to_tensor=True)
            similarity_score = util.cos_sim(ebay_text_embedding, bgs_title_embedding).item()
        candidate_scores.append({'card': candidate_card, 'score': similarity_score})

    # 3. 根据 BERT 相似度得分排序
    ranked_candidates = sorted(candidate_scores, key=lambda x: x['score'], reverse=True)  # 按相似度降序排序

    # 4. 返回排序后的 Top 10 结果 (bgs_title 列表)
    top_10_bgs_titles = [item['card']['_source'].get('bgs_title') for item in
                         ranked_candidates[:10]]  # 取前 10 个结果的 bgs_title
    return top_10_bgs_titles


def get_top_n_es_candidates(ebay_text, n=100):
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
        "size": 100  # 限制返回结果数量为 10
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

    es_query["size"] = n  # 设置返回候选数量为 N
    try:
        res = es.search(index="cards", body=es_query)
        hits = res['hits']['hits']
        return hits  # 返回 hits 列表
    except Exception as e:
        print(f"ES query error: {e}")
        return []


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


ebay_data = pd.read_csv(r"D:\Code\ML\Text\test\ebay_record_2024h2.csv")

example_list = list(ebay_data['name'])
# example_list = [
#
# ]

# 示例用法
for ebay_text_example in example_list:
    # ebay_text_example = "Bryan Bresee 2023 Donruss Optic #276 Purple Shock RC New Orleans Saints"
    top_10_titles = search_card_by_ebay_text(ebay_text_example)
    if top_10_titles:
        print(f"text: '{ebay_text_example}':")
        for title in top_10_titles:
            print(f"- {title}")
            break
    else:
        print(f"No matching cards found for eBay text: '{ebay_text_example}'")
