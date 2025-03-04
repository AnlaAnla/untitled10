import time

from sentence_transformers import SentenceTransformer
import numpy as np
from elasticsearch import Elasticsearch, helpers

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


def vector_search(es_client, index_name, query_vector, top_k=5):
    """执行单个向量搜索。"""

    search_query = {
        "size": top_k,
        "query": {
            "knn": {  # 使用kNN
                "field": "embedding",  # 关键修改： 使用 'field' 参数指定字段名
                "query_vector": query_vector.tolist(),  # 查询向量
                "k": top_k,
                "num_candidates": 50  # 增加候选项
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


if __name__ == '__main__':
    index_name_list = ["program_index", "card_set_index", "athlete_index"]

    t0 = time.time()
    # 示例：单个向量搜索
    query_text = "2022 Panini Contenders Optic - MVP  Orange Prizm #16 Jonathan Taylor /50"
    query_vector = model.encode([query_text], normalize_embeddings=True, convert_to_numpy=True)[0]

    for index_name in index_name_list:
        single_search_results = vector_search(es, index_name, query_vector, top_k=3)
        print(f"{index_name} results:", single_search_results)

    print("--- %s seconds ---" % (time.time() - t0))


