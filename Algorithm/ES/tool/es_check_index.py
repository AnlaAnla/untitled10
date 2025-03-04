from elasticsearch import Elasticsearch

es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# 获取所有索引的信息
response = es.indices.get_alias()

# 打印所有索引的名称
for index_name in response:
    print(index_name)

# 或者，更详细地打印每个索引的别名信息
for index_name, alias_info in response.items():
    print(f"Index: {index_name}")
    if alias_info['aliases']:  # 检查是否有别名
        print("  Aliases:")
        for alias_name in alias_info['aliases']:
            print(f"    - {alias_name}")
    else:
        print("No aliases")

print('end')
