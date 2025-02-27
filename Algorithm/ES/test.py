from elasticsearch import Elasticsearch

es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])


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

program_index_name = "program_index"
card_set_index_name = "card_set_index"
athlete_index_name = "athlete_index"

# 2020 Panini Contenders Draft Picks #77 Noah Fent Iowa hawkeyes football card
# 模型输出:  {'year': '2020', 'program': 'Contenders Draft Picks', 'card_set': '#77', 'card_num': '', 'athlete': 'Noah Fent'}

text = "#77"
# print(program_index_name, es_fuzzy_search(es, index_name=program_index_name, field_name='program', query=text))
print(card_set_index_name, es_fuzzy_search(es, index_name=card_set_index_name, field_name='card_set', query=text))
# print(athlete_index_name, es_fuzzy_search(es, index_name=athlete_index_name, field_name='athlete', query=text, query_type="match_phrase"))
