from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import time


def batch_search_vec2text(texts, tag_list, vec_data, name_list, model, top_k=5):
    """
    批量搜索文本对应的向量，并返回最相似的 top_k 个结果。

    Args:
        texts: 待查询文本列表。
        vec_data: 向量库数据。
        name_list: 向量库对应的名称列表。
        model: SentenceTransformer 模型。
        top_k: 返回最相似结果的数量。

    Returns:
        一个列表，每个元素是一个字典，包含查询文本和对应的 top_k 个结果。
    """

    # 1. 批量编码文本
    output_vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    # 2. 批量计算余弦相似度
    #   output_vecs:  (num_queries, embedding_dim)
    #   vec_data:     (num_vectors_in_db, embedding_dim)
    #   similarities: (num_queries, num_vectors_in_db)
    similarities = np.dot(output_vecs, vec_data.T)  # 利用矩阵乘法高效计算

    results = []
    for i, text in enumerate(texts):
        sim_scores = similarities[i]

        # 3. 获取 top_k 结果
        if len(sim_scores) < top_k:
            top_k_indices = np.argsort(sim_scores)[::-1]  # 全部倒序
        else:
            top_k_indices = np.argpartition(sim_scores, -top_k)[-top_k:]  # 最快的topk
            top_k_indices = top_k_indices[np.argsort(sim_scores[top_k_indices])][::-1]  # topk倒序

        top_k_names = [str(name_list[j]) for j in top_k_indices]
        top_k_scores = [float(sim_scores[j]) for j in top_k_indices]

        # 将结果组合成字典
        result_dict = {
            "query": text,
            "match_tag": tag_list[i],
            "top_k_results": list(zip(top_k_names, top_k_scores))
        }
        results.append(result_dict)

    return results


def test_accuracy(batch_results):
    data_length = len(batch_results)
    top_n = [0, 0, 0, 0, 0]

    for result in batch_results:
        print(f"[{result['query']}] : [{result['match_tag']}]")

        for i, (name, score) in enumerate(result['top_k_results']):
            if result['match_tag'].lower().strip() == name.lower().strip():
                top_n[i] += 1
                if i == 0:
                    print(f"Top1 💖 : {name} [{score}]")
                else:
                    print(f"Top{i + 1} ❤ : {name} [{score}]")
                break

            print(f"--{name} [{score}]")
            if i == 4:
                print('❌')

        print()

    for i in range(1, len(top_n)):
        top_n[i] = top_n[i] + top_n[i - 1]

    for i in range(len(top_n)):
        print(f"Top {i + 1} accuracy: {top_n[i] / data_length * 100:.2f}%, {top_n[i]}/{data_length}")


def test_tag(tag, tag_vec_name):
    vec_data = np.load(f"../temp/{tag_vec_name}_vec.npy")
    name_list = np.load(f"../temp/{tag_vec_name}_vec_names.npy")
    print(f'加载 {tag} 向量库和名称库')

    t1 = time.time()
    # 批量搜索
    test_tag_list = test_data[tag]
    batch_results = batch_search_vec2text(ebay_text_list, test_tag_list, vec_data, name_list, model, top_k=5)

    print('time cost:', time.time() - t1)

    # 打印结果
    test_accuracy(batch_results)
    print(f"----------- {tag} --------------")


if __name__ == '__main__':
    # 加载微调后的模型
    model = SentenceTransformer(r"D:\Code\ML\Model\huggingface\all-MiniLM-L6-v2_fine_tag5")

    test_data = pd.read_excel(r"D:\Code\ML\Text\embedding\ebay_2023_data01_test2.xlsx")
    ebay_text_list = test_data["ebay_text"]

    # test_tag("program", "program")
    test_tag("card_set", "cardSet")
    # test_tag("athlete", "athlete")
