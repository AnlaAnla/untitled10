from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd


def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度（vec1 为单个向量，vec2 可以是向量集合）。"""
    vec1 = vec1.reshape(1, -1)  # 确保 vec1 是行向量
    dot_product = np.dot(vec2, vec1.T)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2, axis=1, keepdims=True)
    similarities = dot_product / (norm_vec1 * norm_vec2)
    return similarities.flatten()


def add_text2vector(batch_texts):
    global text_id, vec_data_list, name_list  # 使用列表 vec_data_list

    output_vecs = model.encode(batch_texts, normalize_embeddings=True)  # 批量生成向量

    text_id += len(batch_texts)
    name_list.extend(batch_texts)
    vec_data_list.extend(output_vecs.tolist())  # 将向量添加到列表中
    print(f"添加向量: {text_id - len(batch_texts) + 1} 到 {text_id}")


def process_text(data_path, save_name):
    global text_id, vec_data_list, name_list  # 使用列表 vec_data_list
    vec_data_list = []  # 初始化为空列表
    name_list = []
    text_id = 0

    # 开始存储向量
    # data_path = r"D:\Code\ML\Text\checklist_tags\cardSet_tags.txt"
    data_list = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            tag = line.strip()
            if tag:  # 排除空行
                data_list.append(tag)
    print('length: ', len(data_list))

    batch_size = 8000
    for i in range(0, len(data_list), batch_size):
        batch_texts = data_list[i:i + batch_size]
        add_text2vector(batch_texts)

    # 将 vec_data_list 转换为 NumPy 数组
    vec_data = np.array(vec_data_list)

    np.save(f"temp/{save_name}_vec.npy", vec_data)
    np.save(f"temp/{save_name}_vec_names.npy", np.array(name_list))  # 也将 name_list 转换为 NumPy 数组
    print(f'{save_name} 向量库建立完成')

    vec_data_list = []  # 初始化为空列表
    name_list = []
    text_id = 0

if __name__ == '__main__':
    vec_data_list = []  # 初始化为空列表
    name_list = []
    text_id = 0

    # 加载模型
    model = SentenceTransformer(r"D:\Code\ML\Model\huggingface\all-MiniLM-L6-v2_fine_tag5")
    # model = SentenceTransformer('BAAI/bge-large-en-v1.5')

    process_text(r"D:\Code\ML\Text\checklist_tags\2023\program.txt", save_name="program")
    process_text(r"D:\Code\ML\Text\checklist_tags\2023\card_set.txt", save_name="cardSet")
    process_text(r"D:\Code\ML\Text\checklist_tags\2023\athlete.txt", save_name="athlete")


