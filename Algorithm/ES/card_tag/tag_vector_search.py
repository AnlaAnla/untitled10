import time
import itertools
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch, helpers
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mysite.settings")

# Elasticsearch 连接配置
es_config = {
    "hosts": [{"host": "localhost", "port": 9200, "scheme": "http"}],  # 或 "https" 如果启用了 SSL
    # 如果需要用户名和密码
    # "http_auth": ("your_username", "your_password")
}

# 创建 Elasticsearch 客户端
es = Elasticsearch(**es_config)


# def vector_search(es_client, index_name, query_vector, top_k=5):
#     """执行单个向量搜索。"""
#
#     search_query = {
#         "size": top_k,
#         "query": {
#             "knn": {  # 使用kNN
#                 "field": "embedding",  # 关键修改： 使用 'field' 参数指定字段名
#                 "query_vector": query_vector.tolist(),  # 查询向量
#                 "k": top_k,
#                 "num_candidates": 50  # 增加候选项
#             }
#         }
#     }
#     response = es_client.search(index=index_name, body=search_query)
#     results = []
#     for hit in response['hits']['hits']:
#         results.append({
#             "name": hit['_source']['name'],
#             "score": hit['_score'],
#             # "text_id": hit["_source"]["text_id"] #可选
#         })
#     return results

def vector_search_script_score(es_client, index_name, query_vector, top_k=5):
    """使用 script_score 查询执行向量搜索（点积）。"""

    search_query = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "(dotProduct(params.query_vector, 'embedding') + 1) / 2",
                    "params": {
                        "query_vector": query_vector.tolist()
                    }
                }
            }
        }
    }
    response = es_client.search(index=index_name, body=search_query)
    results = []
    for hit in response['hits']['hits']:
        results.append({
            "name": hit['_source']['name'],
            "score": hit['_score'],
        })
    return results


def combine_results(results_list, top_k_combined=5, weights=None):
    """
    组合来自多个索引的搜索结果（加权平均）。

    Args:
        results_list:  一个列表，每个元素是来自一个索引的搜索结果列表（每个结果是一个字典）。
        top_k_combined:  要返回的组合结果数量。
        weights:  一个可选的权重列表，用于加权平均。如果为 None，则使用每个索引的最高分数作为权重。

    Returns:
        一个列表，包含组合后的结果（字典），按组合分数降序排列。
    """

    if weights is None:
        # 使用每个索引的最高分数作为权重, 避免空列表
        weights = [results[0]['score'] if results else 1.0 for results in results_list]

    # 权重归一
    sum_weight = sum(weights)
    normalized_weights = [w / sum_weight for w in weights]

    # 生成所有可能的组合
    all_combinations = list(itertools.product(*results_list))

    combined_results = []
    for combination in all_combinations:
        combined_score = 0
        combined_names = []
        for i, result in enumerate(combination):
            combined_score += normalized_weights[i] * result['score']  # 加权平均
            combined_names.append(result['name'])

        combined_results.append({
            "program": combined_names[0],  # 组合名称
            "card_set": combined_names[1],
            "athlete": combined_names[2],
            "combined_score": combined_score  # 组合分数
        })

    # 按组合分数降序排序，并返回 top_k 个结果
    combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
    return combined_results[:top_k_combined]


def test_combined_accuracy(model, test_data, index_name_list, top_k_per_index):
    """
    测试组合准确率，并针对不同的 top_k_combined 值进行测试。
    """
    ebay_text_list = test_data["ebay_text"].tolist()
    test_data_length = len(ebay_text_list)
    query_vectors = model.encode(ebay_text_list, normalize_embeddings=True, convert_to_numpy=True)

    top_n_yes = [0] * max_combinations  # 初始化 Top-N 正确计数列表

    for i, query_vector in enumerate(query_vectors):
        # 从每个索引中搜索
        all_results = []
        for index_name in index_name_list:
            single_search_results = vector_search_script_score(es, index_name, query_vector,
                                                               top_k=top_k_per_index)
            all_results.append(single_search_results)

        # 组合结果
        combined_results = combine_results(all_results)

        # 测试前n个的准确数量

        for j in range(5):
            this_combination = combined_results[j] if combined_results else None

            # 检查预测是否正确
            if this_combination and (
                    this_combination["program"].lower().strip() == test_data["program"].iloc[i].lower().strip() and
                    this_combination["card_set"].lower().strip() == test_data["card_set"].iloc[i].lower().strip() and
                    this_combination["athlete"].lower().strip() == test_data["athlete"].iloc[i].lower().strip()
            ):
                top_n_yes[j] += 1
                print(f"💖 Top {j+1}: {this_combination}")
                break
            else:
                print(f"--{this_combination}")

            if j == 4:
                print("❌")
        print()

    for i in range(1, max_combinations):
        top_n_yes[i] += top_n_yes[i - 1]

    for i in range(max_combinations):
        accuracy = top_n_yes[i] / test_data_length * 100
        print(f"Top {i+1} Accuracy: {accuracy:.2f}% ({top_n_yes[i]}/{test_data_length})")


if __name__ == '__main__':
    # SentenceTransformer 模型路径
    model_path = r"D:\Code\ML\Model\huggingface\all-MiniLM-L6-v2_fine_tag5"
    model = SentenceTransformer(model_path)

    index_name_list = ["2023_program_index", "2023_card_set_index", "2023_athlete_index"]
    top_k_per_index = 5  # 每个索引返回的前 k 个结果
    max_combinations = 5  # 最大组合数量

    # test_data = pd.read_excel(r"D:\Code\ML\Text\embedding\ebay_2023_data01_test2.xlsx")
    # test_combined_accuracy(model, test_data, index_name_list, top_k_per_index)

    # 测试单个案例
    ebay_text = "2023-24 Panini Mosaic #6 Stephen Curry Elevate Mosaic Green Warriors"
    query_vector = model.encode(ebay_text, normalize_embeddings=True, convert_to_numpy=True)
    all_results = []
    for index_name in index_name_list:
        single_search_results = vector_search_script_score(es, index_name, query_vector,
                                                           top_k=top_k_per_index)
        all_results.append(single_search_results)
    combined_results = combine_results(all_results, top_k_combined=5)

    print(combined_results)
    print()
