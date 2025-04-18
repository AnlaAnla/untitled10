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
    "hosts": [{"host": "localhost", "port": 9200, "scheme": "http"},],
    "request_timeout": 30
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


def test_combined_accuracy(model, test_data, index_name, top_k_per_index):
    """
    测试组合准确率，并针对不同的 top_k_combined 值进行测试。
    """
    ebay_text_list = test_data["checklist_name"].tolist()
    test_data_length = len(ebay_text_list)
    query_vectors = model.encode(ebay_text_list, normalize_embeddings=True, convert_to_numpy=True)

    top_n_yes = [0] * top_k_per_index  # 初始化 Top-N 正确计数列表

    for i, query_vector in enumerate(query_vectors):
        # 从每个索引中搜索
        single_search_results = vector_search_script_score(es, index_name, query_vector,
                                                           top_k=top_k_per_index)

        # 测试前n个的准确数量
        print(f"{test_data['checklist_name'].iloc[i]} "
              f"[{test_data["ebay_text"].iloc[i]}]")

        for j in range(5):
            this_combination = single_search_results[j] if single_search_results else None

            # 检查预测是否正确
            if this_combination and (
                    this_combination["name"].lower().strip() == test_data["ebay_text"].iloc[i].lower().strip()
            ):
                top_n_yes[j] += 1
                print(f"💖 Top {j + 1}: {this_combination}")
                break
            else:
                print(f"--{this_combination}")

            if j == 4:
                print("❌")
        print()

    for i in range(1, top_k_per_index):
        top_n_yes[i] += top_n_yes[i - 1]

    for i in range(top_k_per_index):
        accuracy = top_n_yes[i] / test_data_length * 100
        print(f"Top {i + 1} Accuracy: {accuracy:.2f}% ({top_n_yes[i]}/{test_data_length})")


def batch_vector_search_msearch(es_client, index_name, query_vectors, top_k=5):
    """使用 _msearch API 执行批量向量搜索."""
    bulk_search_body = []
    for query_vector in query_vectors:
        search_header = {"index": index_name}
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
        bulk_search_body.extend([search_header, search_query])

    responses = es_client.msearch(body=bulk_search_body)
    all_results = []
    for response in responses['responses']:  # 遍历每个子请求的响应
        results = []
        if 'hits' in response:  # 检查是否有 hits
            for hit in response['hits']['hits']:
                results.append({
                    "name": hit['_source']['name'],
                    "score": hit['_score'],
                })
        all_results.append(results)
    return all_results


def test_combined_accuracy_batch_chunked(model, test_data, index_name, top_k_per_index, batch_size=50):
    """
    使用批量搜索和 Chunking 测试组合准确率，处理大型数据集.
    """
    ebay_text_list = test_data["checklist_name"].tolist()
    test_data_length = len(ebay_text_list)
    query_vectors = model.encode(ebay_text_list, normalize_embeddings=True, convert_to_numpy=True)

    top_n_yes = [0] * top_k_per_index  # 初始化 Top-N 正确计数列表
    total_processed = 0

    for i in range(0, test_data_length, batch_size):  # 循环处理数据批次
        batch_ebay_texts = ebay_text_list[i:i + batch_size]
        batch_query_vectors = query_vectors[i:i + batch_size]

        # 执行批量搜索 (针对当前批次的数据)
        all_batch_results = batch_vector_search_msearch(es, index_name, batch_query_vectors, top_k=top_k_per_index)

        for j in range(len(batch_ebay_texts)):  # 遍历当前批次的结果
            single_batch_results = all_batch_results[j]
            query_index_in_full_data = i + j  # 计算当前查询在完整数据集中的索引

            # 测试前n个的准确数量
            print(f"{test_data['checklist_name'].iloc[query_index_in_full_data]} "
                  f"[{test_data["ebay_text"].iloc[query_index_in_full_data]}]")

            for k in range(5):
                this_combination = single_batch_results[k] if k < len(single_batch_results) else None

                # 检查预测是否正确
                if this_combination and (
                        this_combination["name"].lower().strip() == test_data["ebay_text"].iloc[
                    query_index_in_full_data].lower().strip()
                ):
                    top_n_yes[k] += 1
                    print(f"💖 Top {k + 1}: {this_combination}")
                    break
                else:
                    print(f"--{this_combination}")

                if k == 4:
                    print("❌")
            print()
            total_processed += 1

    for i in range(1, top_k_per_index):
        top_n_yes[i] += top_n_yes[i - 1]

    for i in range(top_k_per_index):
        accuracy = top_n_yes[i] / test_data_length * 100
        print(f"Top {i + 1} Accuracy: {accuracy:.2f}% ({top_n_yes[i]}/{test_data_length})")
    print(f"Total processed: {total_processed}")  # 验证处理了所有数据


if __name__ == '__main__':
    # SentenceTransformer 模型路径
    model_path = r'D:\Code\ML\Model\huggingface\all-MiniLM-L6-v2_checklist_ebay03'  # 替换为你的模型路径
    model = SentenceTransformer(model_path)

    index_name_list = 'checklist_ebay_vec_data_2023'
    top_k_per_index = 5  # 每个索引返回的前 k 个结果

    test_data = pd.read_excel(r"D:\Code\ML\Text\embedding\checklist_ebay_data_2023\checklist_ebay_data_test.xlsx")
    # test_combined_accuracy(model, test_data, index_name_list, top_k_per_index)
    test_combined_accuracy_batch_chunked(model, test_data, index_name_list, top_k_per_index)

    # 测试单个案例
    # ebay_text = "2023-24 Panini Mosaic #6 Stephen Curry Elevate Mosaic Green Warriors"
    # query_vector = model.encode(ebay_text, normalize_embeddings=True, convert_to_numpy=True)
    # all_results = []
    # for index_name in index_name_list:
    #     single_search_results = vector_search_script_score(es, index_name, query_vector,
    #                                                        top_k=top_k_per_index)
    #     all_results.append(single_search_results)
    # combined_results = combine_results(all_results, top_k_combined=5)
    #
    # print(combined_results)
    # print()


'''
Top 1 Accuracy: 54.41% (6843/12577)
Top 2 Accuracy: 73.64% (9262/12577)
Top 3 Accuracy: 82.69% (10400/12577)
Top 4 Accuracy: 86.90% (10929/12577)
Top 5 Accuracy: 89.50% (11257/12577)
Total processed: 12577
'''
