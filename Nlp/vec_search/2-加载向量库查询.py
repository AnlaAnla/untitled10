from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

def cosine_similarity(vec1, vec2):
    """计算两个已 L2 归一化向量的余弦相似度。"""
    vec2 = vec2.flatten()  # 将vec2展开成1维
    return np.dot(vec1, vec2)


def search_vec2text(text, top_k:int=None):
    """
    :param top_k: 如果为None, 返回最大的, 如果有数字, 返回排行前 top_k的
    """
    global text_id, vec_data, name_list

    output_vec = model.encode(text).reshape(1, -1)
    distances = np.apply_along_axis(cosine_similarity, 1, vec_data, output_vec)

    if top_k is None:
        max_dis = np.max(distances)
        search_name = name_list[np.argmax(distances)]

        print(search_name, ': ', max_dis)
        return search_name
    else:
        search_names_dis = []
        if len(distances) < top_k:
            top_k = len(distances)

        # 获取最相似的k个元素
        partitioned_indices = np.argpartition(distances, -top_k)
        top_k_indices_arr = partitioned_indices[-top_k:][np.argsort(distances[partitioned_indices[-top_k:]])][::-1]
        for i in top_k_indices_arr:
            search_names_dis.append([str(name_list[i]), float(distances[i])])

        for item in search_names_dis:
            print(item)
        return search_names_dis



if __name__ == '__main__':

    # 加载模型
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    vec_data = np.load("temp/vec_data_text01.npy")
    name_list = np.load("temp/vec_data_text01_names.npy")
    print('加载向量库, 和名称库')

    print()

