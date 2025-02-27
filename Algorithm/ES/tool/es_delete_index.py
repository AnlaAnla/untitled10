from elasticsearch import Elasticsearch

# 连接到 Elasticsearch (请根据你的 ES 配置修改)
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

index_name = "card_set_index"

if es.indices.exists(index=index_name):
    try:
        es.indices.delete(index=index_name)
        print(f"Index '{index_name}' successfully deleted.")
    except Exception as e:
        print(f"Error deleting index '{index_name}': {e}")
else:
    print(f"Index '{index_name}' does not exist.")