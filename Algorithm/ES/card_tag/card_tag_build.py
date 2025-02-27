from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk  # 导入 bulk helper

# 1. 连接 Elasticsearch (根据你的ES配置修改)
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
if not es.ping():
    raise Exception("Elasticsearch 连接失败")


# 2. 加载标签文件
def load_tags(filepath):
    tags = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            tag = line.strip()
            if tag:  # 排除空行
                tags.append(tag)
    return tags

program_tags = load_tags(r"D:\Code\ML\Text\checklist_tags\program_tags.txt")
card_set_tags = load_tags(r"D:\Code\ML\Text\checklist_tags\cardSet_tags.txt")
athlete_tags = load_tags(r"D:\Code\ML\Text\checklist_tags\athlete_tags.txt")

# 3. 定义索引名称
program_index_name = "program_index"
card_set_index_name = "card_set_index"
athlete_index_name = "athlete_index"


# 4. 创建索引 (如果索引不存在) 和 导入数据
def create_and_populate_index(es_client, index_name, tags, field_name):
    if not es_client.indices.exists(index=index_name):
        print(f"创建索引: {index_name}")
        es_client.indices.create(index=index_name)

    print(f"批量导入数据到索引: {index_name}")
    actions = []  # 存储 bulk 操作的列表
    for tag in tags:
        action = {
            "_index": index_name,
            "_source": {field_name: tag}
        }
        actions.append(action)

    # 使用 bulk helper 批量索引
    success, errors = bulk(es_client, actions)
    print(f"成功索引: {success}, 失败索引: {errors if errors else 0}")
    if errors:
        print("部分文档索引失败，请检查错误信息。")

create_and_populate_index(es, program_index_name, program_tags, "program")
create_and_populate_index(es, card_set_index_name, card_set_tags, "card_set")
create_and_populate_index(es, athlete_index_name, athlete_tags, "athlete")  # 如果需要索引 athlete_tags

es.indices.refresh(index=[card_set_index_name, program_index_name, athlete_index_name])  # 刷新索引，确保数据可以被搜索
print("ES 索引准备完成")
