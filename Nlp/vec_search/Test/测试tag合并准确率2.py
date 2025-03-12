from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import time
import itertools  # 引入 itertools 模块


# 从第一个代码片段复制 combine_results 函数
def combine_results(results_list, top_k_combined=5, weights=None):
    """
    组合来自多个索引的搜索结果（加权平均）。

    Args:
        results_list:  一个列表，每个元素是来自一个索引的搜索结果列表（每个结果是一个字典）。
        top_k_combined:  要返回的组合结果数量。
        weights:  一个可选的权重列表，用于加权平均。如果为 None，则使用每个索引的最高分数作为权重。

    Returns:
        一个列表，包含组合后的结果（字典），按组合分数降序排列。
    """

    if weights is None:
        # 使用每个索引的最高分数作为权重, 避免空列表
        weights = [results[0]['score'] if results else 1.0 for results in results_list]

    # 权重归一
    sum_weight = sum(weights)
    normalized_weights = [w / sum_weight for w in weights]

    # 生成所有可能的组合
    all_combinations = list(itertools.product(*results_list))

    combined_results = []
    for combination in all_combinations:
        combined_score = 0
        combined_names = []
        for i, result in enumerate(combination):
            combined_score += normalized_weights[i] * result['score']  # 加权平均
            combined_names.append(result['name'])

        combined_results.append({
            "program": combined_names[0],  # 组合名称
            "card_set": combined_names[1],
            "athlete": combined_names[2],
            "combined_score": combined_score  # 组合分数
        })

    # 按组合分数降序排序，并返回 top_k 个结果
    combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
    return combined_results[:top_k_combined]


def batch_search_vec2text(texts, vec_data, name_list, model, top_k=5):  # 修改: top_k 默认值为 10，返回 top_k 结果
    """
    批量搜索文本对应的向量，并返回最相似的 top_k 个结果。
    (修改: top_k 默认值为 10, 返回 top_k 结果)

    Args:
        texts: 待查询文本列表。
        vec_data: 向量库数据。
        name_list: 向量库对应的名称列表。
        model: SentenceTransformer 模型。
        top_k: 返回最相似结果的数量。

    Returns:
        一个列表，每个元素是一个结果列表，每个结果是一个字典，包含 'name' 和 'score'。
    """

    # 1. 批量编码文本
    output_vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    # 2. 批量计算余弦相似度
    similarities = np.dot(output_vecs, vec_data.T)

    # 3. 获取 top_k 结果 (修改：返回 top_k)
    top_k_indices = np.argsort(similarities, axis=1)[:, ::-1][:, :top_k]  # 获取 top_k 索引
    top_k_scores = np.take_along_axis(similarities, top_k_indices, axis=1)  # 获取 top_k 分数

    batch_results = []
    for i in range(len(texts)):
        results = []
        for j in range(top_k):
            results.append({'name': str(name_list[top_k_indices[i, j]]), 'score': top_k_scores[i, j]})
        batch_results.append(results)
    return batch_results  # 返回结果列表


def test_combined_accuracy(model, test_data, top_k_per_index=10,
                           max_combinations=5):  # 修改: 添加 top_k_per_index 和 max_combinations 参数
    """
    测试组合准确率：输入 ebay 文本，获取 top_k 的 program, card_set, athlete 的向量搜索结果，
    然后使用 combine_results 组合结果，并测试 top N (N=1 to max_combinations) 的准确率。
    """
    ebay_text_list = test_data["ebay_text"].tolist()
    test_data_length = len(ebay_text_list)
    top_n_yes = [0] * max_combinations  # 初始化 Top-N 正确计数列表

    # 1. 分别获取三个 tag 的 top_k 预测结果
    program_results = batch_search_vec2text(ebay_text_list, *load_vec_data("program"), model, top_k=top_k_per_index)
    card_set_results = batch_search_vec2text(ebay_text_list, *load_vec_data("cardSet"), model, top_k=top_k_per_index)
    athlete_results = batch_search_vec2text(ebay_text_list, *load_vec_data("athlete"), model, top_k=top_k_per_index)

    for i in range(test_data_length):
        # 准备当前样本的搜索结果列表给 combine_results 函数
        current_results_list = [
            program_results[i],
            card_set_results[i],
            athlete_results[i]
        ]

        # 组合结果
        combined_results = combine_results(current_results_list, top_k_combined=max_combinations)

        # 测试前n个的准确数量
        print(f"{test_data['ebay_text'].iloc[i]} "
              f"[{test_data["program"].iloc[i]}, {test_data["card_set"].iloc[i]}, {test_data["athlete"].iloc[i]}]")

        for j in range(max_combinations):
            this_combination = combined_results[j] if combined_results and len(
                combined_results) > j else None  # 确保索引不越界

            # 检查预测是否正确
            if this_combination and (
                    this_combination["program"].lower().strip() == test_data["program"].iloc[i].lower().strip() and
                    this_combination["card_set"].lower().strip() == test_data["card_set"].iloc[i].lower().strip() and
                    this_combination["athlete"].lower().strip() in test_data["athlete"].iloc[i].lower().strip()
            ):
                top_n_yes[j] += 1
                print(f"💖 Top {j + 1}: {this_combination}")
                break
            else:
                print(f"--{this_combination}")

            if j == max_combinations - 1:  # 最后一个 top k 也没找到
                print("❌")
        print()

    for i in range(1, max_combinations):
        top_n_yes[i] += top_n_yes[i - 1]

    for i in range(max_combinations):
        accuracy = top_n_yes[i] / test_data_length * 100
        print(f"Top {i + 1} Accuracy: {accuracy:.2f}% ({top_n_yes[i]}/{test_data_length})")
    return top_n_yes  # 返回 top_n_yes 列表，方便进一步分析


def load_vec_data(tag_vec_name):
    """加载向量数据和名称列表"""
    vec_data = np.load(f"../temp/{tag_vec_name}_vec.npy")
    name_list = np.load(f"../temp/{tag_vec_name}_vec_names.npy")
    print(f'加载 {tag_vec_name} 向量库和名称库')
    return vec_data, name_list


if __name__ == '__main__':
    # 加载微调后的模型
    model = SentenceTransformer(r"D:\Code\ML\Model\huggingface\all-MiniLM-L6-v2_fine_tag7")
    # model = SentenceTransformer(r"D:\Code\ML\Model\huggingface\all-mpnet-base-v2_fine_tag01")

    test_data = pd.read_excel(r"D:\Code\ML\Text\embedding\ebay_2023_data01_test.xlsx")
    ebay_text_list = test_data["ebay_text"]

    # test_tag("program", "program")  # 注释掉原来的单个 tag 测试
    # test_tag("card_set", "cardSet")
    # test_tag("athlete", "athlete")

    top_k_per_index = 5  # 设置每个索引搜索的 top_k 值
    max_combinations = 5  # 设置组合结果测试的 top_k 值 (Top-N 中的 N)
    top_n_counts = test_combined_accuracy(model, test_data, top_k_per_index, max_combinations)  # 调用新的组合准确率测试函数, 并传递参数

    print(f"Top N Counts: {top_n_counts}")  # 打印 Top-N 正确数量，方便进一步分析
