from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import time


def batch_search_vec2text(texts, vec_data, name_list, model, top_k=1):
    """
    批量搜索文本对应的向量，并返回最相似的 top_k 个结果。
    (修改: top_k 默认值为 1)

    Args:
        texts: 待查询文本列表。
        vec_data: 向量库数据。
        name_list: 向量库对应的名称列表。
        model: SentenceTransformer 模型。
        top_k: 返回最相似结果的数量。

    Returns:
        一个列表，每个元素是一个字符串，表示最匹配的名称。
    """

    # 1. 批量编码文本
    output_vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    # 2. 批量计算余弦相似度
    similarities = np.dot(output_vecs, vec_data.T)

    # 3. 获取 top_k 结果 (修改：只取 top1)
    top_k_indices = np.argmax(similarities, axis=1)  # 直接获取每行最大值的索引
    top_k_names = [str(name_list[i]) for i in top_k_indices]

    return top_k_names  # 直接返回 top1 名称列表


def test_combined_accuracy(model, test_data):
    """
    测试组合准确率：输入 ebay 文本，获取 top1 的 program, card_set, athlete 的向量搜索结果，
    当这三个同时正确的时候，才计算为正确。
    """
    ebay_text_list = test_data["ebay_text"].tolist()

    # 1. 分别获取三个 tag 的 top1 预测结果
    program_preds = batch_search_vec2text(ebay_text_list, *load_vec_data("program"), model)
    card_set_preds = batch_search_vec2text(ebay_text_list, *load_vec_data("cardSet"), model)
    athlete_preds = batch_search_vec2text(ebay_text_list, *load_vec_data("athlete"), model)

    # 2. 计算组合准确率
    correct_count = 0
    total_count = len(test_data)

    for i in range(total_count):
        if (
                program_preds[i].lower().strip() == test_data["program"].iloc[i].lower().strip() and
                card_set_preds[i].lower().strip() == test_data["card_set"].iloc[i].lower().strip() and
                athlete_preds[i].lower().strip() == test_data["athlete"].iloc[i].lower().strip()
        ):
            correct_count += 1

    accuracy = correct_count / total_count * 100
    print(f"Combined Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")
    return accuracy


def load_vec_data(tag_vec_name):
    """加载向量数据和名称列表"""
    vec_data = np.load(f"../temp/{tag_vec_name}_vec.npy")
    name_list = np.load(f"../temp/{tag_vec_name}_vec_names.npy")
    print(f'加载 {tag_vec_name} 向量库和名称库')
    return vec_data, name_list


if __name__ == '__main__':
    # 加载微调后的模型
    model = SentenceTransformer(r"D:\Code\ML\Model\huggingface\all-MiniLM-L6-v2_fine_tag5")

    test_data = pd.read_excel(r"D:\Code\ML\Text\embedding\ebay_2023_data01_test2.xlsx")
    ebay_text_list = test_data["ebay_text"]

    # test_tag("program", "program")  # 注释掉原来的单个 tag 测试
    # test_tag("card_set", "cardSet")
    # test_tag("athlete", "athlete")

    test_combined_accuracy(model, test_data)  # 调用新的组合准确率测试函数
