import pandas as pd
import re

def create_regex_string_from_list(name_list):
    """
    将字符串列表转换为正则表达式的 "或" 模式字符串，并处理特殊字符。
    """
    escaped_names = [re.escape(name) for name in name_list if isinstance(name, str)] # 确保是字符串类型, 并转义特殊字符
    return "(" + "|".join(escaped_names) + ")" if escaped_names else None

def get_card_set_program_regex_from_csv(csv_file_path, card_set_col='card_set', program_new_col='program_new'):
    """
    从 CSV 文件中读取 'card_set' 和 'program_new' 列，获取去重值，并生成用于正则匹配的字符串。

    Args:
        csv_file_path (str): CSV 文件路径。
        card_set_col (str): card_set 列名，默认为 'card_set'。
        program_new_col (str): program_new 列名，默认为 'program_new'。

    Returns:
        tuple: 包含 card_set 和 program_new 的正则表达式字符串的元组 (card_set_regex_str, program_new_regex_str)。
               如果列不存在或没有有效数据，则对应的字符串为 None。
    """
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at path: {csv_file_path}")
        return None, None

    card_set_unique_values = []
    program_new_unique_values = []

    if card_set_col in df.columns:
        card_set_unique_values = df[card_set_col].dropna().unique()
    else:
        print(f"Warning: Column '{card_set_col}' not found in CSV file.")

    if program_new_col in df.columns:
        program_new_unique_values = df[program_new_col].dropna().unique()
        print(program_new_unique_values)
    else:
        print(f"Warning: Column '{program_new_col}' not found in CSV file.")

    card_set_regex_str = create_regex_string_from_list(card_set_unique_values)
    program_new_regex_str = create_regex_string_from_list(program_new_unique_values)

    return card_set_regex_str, program_new_regex_str


# 示例用法
csv_file = r"D:\Code\ML\Text\test\paniniamerica_checklist_refresh.csv" # 替换为你的 CSV 文件路径
card_set_regex, program_new_regex = get_card_set_program_regex_from_csv(csv_file)

if card_set_regex:
    print("Card Set Regex String:")
    print(card_set_regex)
else:
    print("No valid 'card_set' data found to create regex.")

if program_new_regex:
    print("\nProgram New Regex String:")
    print(program_new_regex)
else:
    print("No valid 'program_new' data found to create regex.")