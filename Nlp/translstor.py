import json
import re
import re
import html
from urllib import parse
import requests

GOOGLE_TRANSLATE_URL = 'http://translate.google.com/m?q=%s&tl=%s&sl=%s'



def translate(text, to_language="auto", text_language="auto"):
    text = parse.quote(text)
    url = GOOGLE_TRANSLATE_URL % (text, to_language, text_language)
    response = requests.get(url)
    data = response.text
    expr = r'(?s)class="(?:t0|result-container)">(.*?)<'
    result = re.findall(expr, data)
    if (len(result) == 0):
        return ""

    return html.unescape(result[0])

def contains_chinese(sentence):
    pattern = re.compile(r'[\u4e00-\u9fff]+')
    return bool(pattern.search(sentence))


def contains_japanese(sentence):
    pattern = re.compile(r'[\u3040-\u30FF\u31F0-\u31FF\uFF65-\uFF9F]+')
    return bool(pattern.search(sentence))


def contains_traditional_chinese(sentence):
    pattern = re.compile(r'[\u2E80-\u2FDF\u3400-\u4DBF]+')
    return bool(pattern.search(sentence))


def contains_english(sentence):
    pattern = re.compile(r'[a-zA-Z]+')
    return bool(pattern.search(sentence))


def contains_number(sentence):
    pattern = re.compile(r'\d+')
    return bool(pattern.search(sentence))


with open(r"D:\Code\ML\texts\ManualTransFile.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

data_list = list(data)

for i in range(50):
    text = data_list[i]
    if (contains_japanese(data_list[i])
            or contains_traditional_chinese(data_list[i])
            or contains_chinese(data_list[i])):

        print(i, text, ' :', translate(text, "zh-CN", "ja"))  # 日语转汉语








