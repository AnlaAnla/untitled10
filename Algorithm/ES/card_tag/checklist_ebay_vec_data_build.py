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

# SentenceTransformer 模型 (与之前相同)吗


# 根据你的模型和数据确定
VECTOR_DIMENSION = 384  # 示例维度，根据你的模型修改


def load_data_from_excel(file_paths):
    """从 Excel 文件加载数据"""
    all_data = []
    for file_path in file_paths:
        df = pd.read_excel(file_path)
        # 确保列名正确
        # df = df.rename(columns={df.columns[0]: 'id', df.columns[1]: 'img', df.columns[2]: 'name'})
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['name'], keep='first')
    return combined_df


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
                        "similarity": "dot_product"
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


def insert_data_to_ES(es_client, index_name, data, batch_size=10000):
    """批量插入数据到 Milvus"""
    num_batches = (len(data) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Inserting data"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))
        batch_data = data[start_idx:end_idx]

        # 提取文本并生成 embeddings
        batch_ebay_id = batch_data['id'].tolist()
        batch_img_url = batch_data['img'].tolist()
        batch_texts = batch_data['name'].tolist()
        batch_embeddings = model.encode(batch_texts, normalize_embeddings=True, convert_to_tensor=True).tolist()

        """将向量数据批量索引到 Elasticsearch。"""
        actions = [
            {
                "_index": index_name,
                "_source": {
                    "ebay_id": batch_ebay_id[j - start_idx],  # 可选
                    "img_url": batch_img_url[j - start_idx],
                    "embedding": batch_embeddings[j - start_idx],  # 转换为列表
                    "name": batch_texts[j - start_idx],
                }
            }
            for j in range(start_idx, end_idx)
        ]
        success, _ = helpers.bulk(es_client, actions)  # 默认500批处理
        print(f"{start_idx}-{end_idx} ,Indexed {success} documents into '{index_name}'.")


def process_and_index(data_path_list, index_name):
    """读取文本数据，生成向量，并将其索引到 Elasticsearch。"""

    data = load_data_from_excel(data_path_list)

    # 创建索引（如果需要）
    create_index(es, index_name)

    # 将数据索引到 Elasticsearch
    insert_data_to_ES(es, index_name=index_name, data=data)


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
            "ebay_id": hit['_source']['ebay_id'],
            "img_url": hit['_source']['img_url'],
            "name": hit['_source']['name'],
            "score": hit['_score'],
        })
    return results


def batch_vector_search(es_client, index_name, query_texts, top_k=5):
    """执行批量向量搜索。"""
    query_vectors = model.encode(query_texts, normalize_embeddings=True, convert_to_numpy=True)
    results = []

    for i, query_text in enumerate(query_texts):
        query_vector = query_vectors[i]
        search_results = vector_search_script_score(es_client, index_name, query_vector, top_k)
        results.append({
            "query": query_text,
            "top_k_results": search_results
        })

    return results


if __name__ == '__main__':
    model_path = r'D:\Code\ML\Model\huggingface\all-MiniLM-L6-v2_checklist_ebay03'  # 替换为你的模型路径
    model = SentenceTransformer(model_path)

    # 将数据索引到 Elasticsearch
    file_paths = [r"D:\Code\ML\Text\embedding\checklist_ebay_data_2023\ebay_2023_01.xlsx",
                  r"D:\Code\ML\Text\embedding\checklist_ebay_data_2023\ebay_2023_02.xlsx"]  # 文件路径列表

    process_and_index(file_paths, "checklist_ebay_vec_data_2023")

    # # 示例：单个向量搜索
    query_text = "CAITLIN CLARK 2023 Bowman University Chrome 2007-08 Bowman Refractor PSA 9 Mint"
    query_vector = model.encode([query_text], normalize_embeddings=True, convert_to_numpy=True)[0]
    single_search_results = vector_search_script_score(es, "checklist_ebay_vec_data_2023", query_vector, top_k=5)
    print("\nSingle search results:", single_search_results)

    # # 示例：批量向量搜索
    # query_texts = ["query text 1", "query text 2", "query text 3"]
    # batch_search_results = batch_vector_search(es, "program_index", query_texts, top_k=5)
    # print("\nBatch search results:", batch_search_results)
    # for i in batch_search_results:
    #     print(i)
