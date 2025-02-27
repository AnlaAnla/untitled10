from elasticsearch import Elasticsearch

es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

program_index_name = "program_index"
card_set_index_name = "card_set_index"
athlete_index_name = "athlete_index"


def llm_extract(ebay_text):
    if "Donruss Optic" in ebay_text:
        return {
            "program": "Donruss Optic",  # 大模型提取的卡牌系列
            "card_set": "Legendary Logos Holo Prizm",  # 大模型提取的卡种
            "athlete": "Jared Goff",  # 大模型提取的球员名称
            "year": "2021"  # 大模型提取的年份
        }
    elif "MOSAIC" in ebay_text:
        return {
            "program": "PANINI MOSAIC",
            "card_set": "#144",
            "athlete": "REGGIE BUSH",
            "year": "2021"
        }
    elif "Pro Set Metal" in ebay_text:
        return {
            "program": "Pro Set Metal",
            "card_set": "PPP Crystal Snakeskin Autograph 1/1",
            "athlete": "Jelani Woods",
            "year": "2022"
        }
    elif "Panini Prizm" in ebay_text:
        return {
            "program": "Panini Prizm",
            "card_set": "Rookies #333 (RC)",
            "athlete": "Dontayvion Wicks",
            "year": "2023"
        }
    else:
        return {
            "program": None,
            "card_set": None,
            "athlete": None,
            "year": None
        }


def es_fuzzy_search(es_client, index_name, field_name, query, fuzziness="AUTO", query_type="match"):
    """
    在 ES 中进行模糊搜索，可以指定查询类型 (match 或 match_phrase)
    """
    if query_type == "match_phrase":
        search_query = {
            "match_phrase": {
                field_name: {
                    "query": query,
                    "slop": 2,  # 允许最多 2 个单词的间隔 (可以理解为词序错乱或中间插入单词)
                }
            }
        }
    else:  # 默认使用 match 查询
        search_query = {
            "match": {
                field_name: {
                    "query": query,
                    "fuzziness": fuzziness,
                    "prefix_length": 2
                }
            }
        }

    res = es_client.search(index=index_name, query=search_query)
    hits = res['hits']['hits']
    if hits:
        return hits[0]['_source'][field_name], hits[0]['_score']  # 返回最佳匹配的标签和分数
    return None, 0.0


def extract_card_info_with_es(llm_output):
    """
    结合 ES 进行二次校验和信息抽取
    """
    program_llm = llm_output.get("program")
    card_set_llm = llm_output.get("card_set")
    athlete_llm = llm_output.get("athlete")
    print(f"program_llm: {program_llm}, card_set_llm: {card_set_llm}, athlete_llm: {athlete_llm}")

    # --- 卡牌系列校验 ---
    program_es, program_score = es_fuzzy_search(es, program_index_name, "program", query=program_llm)
    if program_es and program_score > 1.0:  # 可以根据实际情况调整分数阈值
        confirmed_program = program_es
    else:
        confirmed_program = program_llm if program_llm else ""

    # --- 卡种校验 ---
    card_set_es, type_score = es_fuzzy_search(es, card_set_index_name, "card_set", query=card_set_llm)  # 卡种可以尝试更强的模糊匹配
    if card_set_es and type_score > 1.0:  # 调整卡种的分数阈值
        confirmed_card_set = card_set_es
    else:
        confirmed_card_set = card_set_llm if card_set_llm else ""

    # --- 球员名称校验 (使用 match_phrase 和 slop=2) ---
    player_name_es, player_score = es_fuzzy_search(es, athlete_index_name, "athlete", query=athlete_llm,
                                                   query_type="match_phrase")
    if player_name_es and player_score > 5.0:  # 调整球员名称的分数阈值，match_phrase 分数通常更高
        confirmed_player_name = player_name_es
    else:
        confirmed_player_name = athlete_llm if athlete_llm else ""

    return {
        "program": confirmed_program,
        "card_set": confirmed_card_set,
        "athlete": confirmed_player_name,
    }


# 测试 ES 二次校验 (保持不变)
ebay_texts = [
    "2021 Donruss Optic Legendary Logos Holo Prizm Jared Goff #LL-7 Detroit Lions",
    "2021 PANINI MOSAIC #144 REGGIE BUSH NEW ORLEANS SAINTS FOOTBALL",
    "2022 Pro Set Metal PPP Crystal Snakeskin Autograph 1/1 Jelani Woods",
    "2023 Panini Prizm - Rookies #333 Dontayvion Wicks (RC)",
    "2022 Pro Set Metal PPP Crystal Snakeskin Autograph 1/1 Jelani  Wood",  # 球员名字略有错误
    "2021 PANINI MOSAIC #144 REGGIE  BUSH NEW ORLEANS SAINTS FOOTBALL",  # 球员名字略有错误
    "2021 Donruss Optic Legendary Logos Holo Prizm Jarod Goff #LL-7 Detroit Lions",  # 球员名字拼写错误
]

for text in ebay_texts:
    llm_output = llm_extract(text)  # 确保你的 llm_extract 函数能正常工作
    extracted_info = extract_card_info_with_es(llm_output)
    print(f"Ebay Text: {text}")
    print(f"LLM Output: {llm_output}")
    print(f"ES Enhanced Output: {extracted_info}")
    print("-" * 30)
