from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd


def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度（vec1 为单个向量，vec2 可以是向量集合）。"""
    vec1 = vec1.reshape(1, -1)  # 确保 vec1 是行向量
    dot_product = np.dot(vec2, vec1.T)  # dot_product shape: (807566, 1)
    norm_vec1 = np.linalg.norm(vec1)  # norm_vec1 shape: (1,)
    norm_vec2 = np.linalg.norm(vec2, axis=1, keepdims=True)  # norm_vec2 shape: (807566, 1)
    similarities = dot_product / (
                norm_vec1 * norm_vec2)  # dot_product 和 (norm_vec1 * norm_vec2) 都是 (807566, 1), 可以直接做除法
    return similarities.flatten()


def search_vec2text(text, alpha=0.2, top_k: int = None):
    """
    :param text: 查询的文本。
    :param alpha: 长度惩罚系数。
    :param top_k: 如果为 None，返回最相似的；如果为整数，返回前 top_k 个最相似的。
    """
    global text_id, vec_data, name_list

    output_vec = model.encode(text).reshape(1, -1)

    # 使用改进后的 cosine_similarity 函数
    similarities = cosine_similarity(output_vec, vec_data)

    # 引入长度惩罚
    for i, name in enumerate(name_list):
        similarities[i] *= (1 - alpha * abs(len(text) - len(name)) / max(len(text), len(name)))

    if top_k is None:
        max_dis = np.max(similarities)
        search_name = name_list[np.argmax(similarities)]

        print(search_name, ': ', max_dis)
        return search_name
    else:
        search_names_dis = []
        if len(similarities) < top_k:
            top_k = len(similarities)

        # 获取最相似的k个元素
        top_k_indices_arr = np.argpartition(similarities, -top_k)[-top_k:]
        top_k_indices_arr = top_k_indices_arr[np.argsort(similarities[top_k_indices_arr])][::-1]
        for i in top_k_indices_arr:
            search_names_dis.append([str(name_list[i]), float(similarities[i])])

        for item in search_names_dis:
            print(item)
        return search_names_dis


if __name__ == '__main__':
    # 加载模型
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    vec_data = np.load("temp/vec_data_text01.npy")
    name_list = np.load("temp/vec_data_text01_names.npy")
    print('加载向量库和名称库')

    # 示例
    search_vec2text("prizm base coby white", top_k=5)
    print()
