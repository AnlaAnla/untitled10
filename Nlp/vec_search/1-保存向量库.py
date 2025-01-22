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

    output_vecs = model.encode(batch_texts)  # 批量生成向量

    text_id += len(batch_texts)
    name_list.extend(batch_texts)
    vec_data_list.extend(output_vecs.tolist())  # 将向量添加到列表中
    print(f"添加向量: {text_id - len(batch_texts) + 1} 到 {text_id}")


if __name__ == '__main__':

    # 加载模型
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    vec_data_list = []  # 初始化为空列表
    name_list = []
    text_id = 0

    # 开始存储向量
    data = pd.read_csv(r"D:\Code\ML\Text\test\paniniamerica_checklist_refresh.csv")
    print('length: ', len(data))

    batch_size = 8000  # 设置 batch size
    for i in range(0, len(data), batch_size):
        batch_texts = data["bgs_title"][i:i + batch_size].tolist()
        add_text2vector(batch_texts)

    # 将 vec_data_list 转换为 NumPy 数组
    vec_data = np.array(vec_data_list)

    np.save("temp/vec_data_text02.npy", vec_data)
    np.save("temp/vec_data_text02_names.npy", np.array(name_list))  # 也将 name_list 转换为 NumPy 数组
    print('向量库建立完成')
