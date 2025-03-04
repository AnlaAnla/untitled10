from sentence_transformers import SentenceTransformer
import numpy as np
from elasticsearch import Elasticsearch, helpers
import pandas as pd
from tqdm import tqdm

# Elasticsearch 连接配置
es_config = {
    "hosts": [{"host": "localhost", "port": 9200, "scheme": "http"}],  # 或 "https" 如果启用了 SSL
    # 如果需要用户名和密码
    # "http_auth": ("your_username", "your_password")
}

# 创建 Elasticsearch 客户端
es = Elasticsearch(**es_config)

# SentenceTransformer 模型 (与之前相同)
model_path = r"D:\Code\ML\Model\huggingface\all-MiniLM-L6-v2_fine_tag5"  # 替换为你的模型路径
model = SentenceTransformer(model_path)

# 根据你的模型和数据确定
VECTOR_DIMENSION = 384  # 示例维度，根据你的模型修改


def create_index(es_client, index_name):
    """创建 Elasticsearch 索引（如果不存在）。"""
    if not es_client.indices.exists(index=index_name):
        index_settings = {
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "dense_vector",
                        "dims": VECTOR_DIMENSION,
                        "index": "true",  # 使用默认的 HNSW 索引
                        "similarity": "cosine"  # 余弦相似度
                    },
                    "name": {"type": "keyword"},
                    "text_id": {"type": "integer"}  # 可选
                }
            }
        }
        es_client.indices.create(index=index_name, body=index_settings)
        print(f"Index '{index_name}' created.")
    else:
        print(f"Index '{index_name}' already exists.")


def index_data(es_client, index_name, vec_data, name_list):
    """将向量数据批量索引到 Elasticsearch。"""
    actions = [
        {
            "_index": index_name,
            "_source": {
                "embedding": vec_data[i].tolist(),  # 转换为列表
                "name": name_list[i],
                "text_id": i  # 可选
            }
        }
        for i in range(len(vec_data))
    ]
    success, _ = helpers.bulk(es_client, actions)  # 默认500批处理
    print(f"Indexed {success} documents into '{index_name}'.")


def process_and_index(data_path, index_name):
    """读取文本数据，生成向量，并将其索引到 Elasticsearch。"""
    data_list = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            tag = line.strip()
            if tag:
                data_list.append(tag)
    print(f"Loaded {len(data_list)} items from {data_path}")

    # 批量生成向量
    vec_data = model.encode(data_list, normalize_embeddings=True, convert_to_numpy=True)
    name_list = np.array(data_list)

    # 创建索引（如果需要）
    create_index(es, index_name)

    # 将数据索引到 Elasticsearch
    index_data(es, index_name, vec_data, name_list)


def vector_search(es_client, index_name, query_vector, top_k=5):
    """执行单个向量搜索。"""

    search_query = {
        "size": top_k,
        "query": {
            "knn": {  # 使用kNN
                "embedding": {
                    "vector": query_vector.tolist(),  # 查询向量
                    "k": top_k
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
            # "text_id": hit["_source"]["text_id"] #可选
        })
    return results


def batch_vector_search(es_client, index_name, query_texts, top_k=5):
    """执行批量向量搜索。"""
    query_vectors = model.encode(query_texts, normalize_embeddings=True, convert_to_numpy=True)
    results = []

    for i, query_text in enumerate(query_texts):
        query_vector = query_vectors[i]
        search_results = vector_search(es_client, index_name, query_vector, top_k)
        results.append({
            "query": query_text,
            "top_k_results": search_results
        })

    return results


if __name__ == '__main__':
    # 将数据索引到 Elasticsearch
    process_and_index(r"D:\Code\ML\Text\checklist_tags\2023\program.txt", "program_index")
    process_and_index(r"D:\Code\ML\Text\checklist_tags\2023\card_set.txt", "card_set_index")
    process_and_index(r"D:\Code\ML\Text\checklist_tags\2023\athlete.txt", "athlete_index")

    # # 示例：单个向量搜索
    # query_text = "some query text"
    # query_vector = model.encode([query_text], normalize_embeddings=True, convert_to_numpy=True)[0]
    # single_search_results = vector_search(es, "program_index", query_vector, top_k=5)
    # print("\nSingle search results:", single_search_results)
    #
    # # 示例：批量向量搜索
    # query_texts = ["query text 1", "query text 2", "query text 3"]
    # batch_search_results = batch_vector_search(es, "program_index", query_texts, top_k=5)
    # print("\nBatch search results:", batch_search_results)
    # for i in batch_search_results:
    #     print(i)
