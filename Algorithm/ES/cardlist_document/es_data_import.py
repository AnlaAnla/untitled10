from elasticsearch import Elasticsearch, helpers
import pandas as pd
import numpy as np

# 连接到 ES
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

if not es.indices.exists(index="cards"):
    mapping = {
        "properties": {
            "id": {"type": "keyword"},
            "sport": {"type": "keyword"},
            "year": {"type": "integer"},
            "brand": {"type": "keyword"},
            "program_new": {"type": "keyword"},
            "card_set": {"type": "keyword"},
            "athlete": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "team": {"type": "keyword"},
            "position": {"type": "keyword"},
            "card_number": {"type": "keyword"},  # 将 card_number 类型改为 keyword
            "bgs_title": {"type": "text"}
            # ... 其他字段 ...
        }
    }
    es.indices.create(index="cards", mappings=mapping)
    print("Index 'cards' created with new mapping (card_number as keyword).")

# 读取 Excel 数据
df = pd.read_csv(r"D:\Code\ML\Text\test\paniniamerica_checklist_refresh_norepeat.csv")


# 转换数据为 JSON 格式，并准备 bulk 操作
def generate_actions():
    for index, row in df.iterrows():
        doc = row.to_dict()
        # 遍历字典的值，将 NaN 替换为 None
        for key, value in doc.items():
            if pd.isna(value):  # 使用 pandas.isna() 检测 NaN
                doc[key] = None  # 替换为 None

        yield {
            "_index": "cards",  # 你的索引名称
            "_id": doc["id"],  # 使用你的id字段
            "_source": doc,
        }


try:
    # 批量导入
    helpers.bulk(es, generate_actions())
    print("Data successfully imported to Elasticsearch!")  # 添加成功导入的提示
except helpers.BulkIndexError as e:
    print("BulkIndexError occurred!")
    for error in e.errors:  # 遍历 errors 列表, 打印每个错误的详细信息
        print(error)
