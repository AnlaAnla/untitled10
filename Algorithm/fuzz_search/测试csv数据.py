from thefuzz import fuzz
from thefuzz import process
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re

def preprocess_year_num(text):
    year_match = re.search(r'\b(20\d{2})(?:-\d{2})?\b', text)
    num_match = re.search(r'#([A-Za-z0-9\-]+)', text)
    return {
        'year': year_match.group(1) if year_match else "",
        'card_num': num_match.group(1) if num_match else ""
    }

def judge_by_vec_search_list(ebay_text: str, vector_list: list[str], pass_word_list: list[str] = None):
    """
    # 用向量搜索的结果对比原文本
    """
    ebay_text = (ebay_text.replace('-', ' ')
                 .replace('.', ' ')
                 .replace('/', ' ')
                 .replace("'", ' ')
                 .replace("’", ' ')
                 .lower())
    ebay_words = set(ebay_text.split())

    # 使用集合推导一次性转换并存储 pass_word_list
    pass_words = set(word.lower() for word in (pass_word_list or []))  # 处理 None 情况

    # 排序 (只在需要时排序)
    vector_list.sort(key=lambda s: len(s.split()), reverse=True)

    for tag in vector_list:
        temp_tag = tag
        tag = tag.replace('.', ' ').replace("'", ' ').strip()
        tag_words = tag.lower().split()
        # all() 和生成器表达式，简洁高效, 检查生成器表达式中的所有条件是否都为 True。
        if all(
                word in ebay_words
                or (word.rstrip("s") in ebay_words)
                or (word + "s" in ebay_words)  # 处理 prizm 和 prizms 之类的字符
                for word in tag_words if word and word not in pass_words
        ):
            return temp_tag
    return ''
    # return False


def search_tag(ebay_text: str, database_list):
    program_pass_word_list = ['the', 'and']
    cardSet_pass_word_list = ['base', 'and', 'set', '-']

    result_list = []
    for i, database in enumerate(database_list):
        if i == 0:
            pass_word_list = program_pass_word_list
        elif i == 1:
            pass_word_list = cardSet_pass_word_list
        else:
            pass_word_list = None

        matches = process.extract(ebay_text, database, scorer=fuzz.partial_ratio, limit=20)
        match_list = [x[0] for x in matches]
        # print(f"{name_list[i]}: {matches}")
        result = judge_by_vec_search_list(ebay_text=ebay_text,
                                          vector_list=match_list,
                                          pass_word_list=pass_word_list)

        result_list.append(result)
        if i == 0:
            # 移除 program
            ebay_text = re.sub(re.escape(result), '', ebay_text, count=1, flags=re.IGNORECASE)
    for i in range(len(result_list)):
        if not result_list[i]:
            result_list[i] = ''
    print(ebay_text, ': ', result_list)
    return result_list


def ebay_text_parse(ebay_text: str, database_list):
    output = {
        'year': '',
        'program': '',
        'card_set': '',
        'card_num': '',
        'athlete': ''
    }
    year_num_result = preprocess_year_num(ebay_text)
    output['year'] = year_num_result['year']
    output['card_num'] = year_num_result['card_num']

    tag_list = search_tag(ebay_text, database_list)

    output['program'] = tag_list[0]
    output['card_set'] = tag_list[1]
    output['athlete'] = tag_list[2]

    return output


def process_row(i: int, row, database_list: list, name_list: list, results: dict):
    """处理单行数据"""
    t1 = time.time()
    print("=" * 20)
    print('第 ', i)

    ebay_text = row['name']
    LLM_output = ebay_text_parse(ebay_text, database_list) # 传入 database_list, name_list

    results[i] = {
        'year': LLM_output['year'],
        'program': LLM_output['program'],
        'card_set': LLM_output['card_set'],
        'card_num': LLM_output['card_num'],
        'athlete': LLM_output['athlete'],
        'time': time.time() - t1
    }
    print("=" * 20, ' |time: ', results[i]['time'])


def main():
    data_path = r"D:\Code\ML\Project\Card_Text_Parse\Data\program_cardSet_athlete.csv"
    data = pd.read_csv(data_path)
    name_list = ['program', 'card_set', 'athlete']
    database_list = []
    for name in name_list:
        database_list.append(list(data[name].dropna().unique()))

    test_data_csv = pd.read_excel(r"D:\Code\ML\Text\test\ebay_record_2024h2(1).xlsx")

    ebay_text = "2023-24 Panini Donruss Optic Stephen Curry #198 / Base #65 Warriors"
    output = ebay_text_parse(ebay_text, database_list)
    print(output)

    # 使用 ThreadPoolExecutor
    # with ThreadPoolExecutor(max_workers=4) as executor:  # 根据你的CPU核心数调整 max_workers
    #     results = {}
    #     futures = [executor.submit(process_row, i, row, database_list, name_list, results)
    #                for i, row in test_data_csv.head(302).iterrows()]
    #
    #     # 等待所有任务完成 (不需要 as_completed, 因为我们直接处理 results 字典)
    #     for future in as_completed(futures):
    #         future.result()  # 确保获取结果, 即使我们不直接使用它, 这样可以捕获异常
    #
    # # 将结果写回 DataFrame
    # for i, result in results.items():
    #     test_data_csv.at[i, 'year'] = result['year']
    #     test_data_csv.at[i, 'program'] = result['program']
    #     test_data_csv.at[i, 'card_set'] = result['card_set']
    #     test_data_csv.at[i, 'card_num'] = result['card_num']
    #     test_data_csv.at[i, 'athlete'] = result['athlete']
    #
    # test_data_csv.to_excel('temp.xlsx', index=False)
    # print('end')


if __name__ == '__main__':
    main()





