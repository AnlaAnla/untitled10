from elasticsearch import Elasticsearch

# 连接到 Elasticsearch (请根据你的 ES 配置修改)
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# index_name_list = ["2023_program_index", "2023_card_set_index", "2023_athlete_index"]
index_name_list = ['checklist_ebay_vec_data_2023']

for index_name in index_name_list:
    if es.indices.exists(index=index_name):
        try:
            es.indices.delete(index=index_name)
            print(f"Index '{index_name}' successfully deleted.")
        except Exception as e:
            print(f"Error deleting index '{index_name}': {e}")
    else:
        print(f"Index '{index_name}' does not exist.")