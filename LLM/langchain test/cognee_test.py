from openai import OpenAI
from typing import List
import json

client = OpenAI(
    api_key="sk-4395948fb2274a2fb7ef3ac4cc7b8804",
    base_url="https://api.deepseek.com"
)


def count_num(text: str) -> int:
    return len(text.strip())


system_prompt = '''
请遵循以下规则:
1.如果用户问天气相关, 就调用你的工具
2.其他情况正常回答

用json格式回答, 严格遵循以下格式
回答格式: {"text":{正常回答}}
'''

# messages = [
#     {"role": "system", "content": system_prompt},
#     {"role": "user", "content": "我下面说的话有多少个字: 你真是一个大漂亮, 大聪明"}
# ]

messages = [
    {"role": "system", "content": system_prompt},

]


# 1. 你本地真正会执行的函数
def get_weather(city: str, unit: str = "celsius"):
    # 这里先写死，真实项目里你可以请求天气API
    return {
        "city": city,
        "temperature": 25,
        "unit": unit,
        "weather": "rain"
    }

tool_map = {
    "get_weather": get_weather
}


# 2. 告诉模型：你有哪些工具可用
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位"
                    }
                },
                "required": ["city"]
            }
        }
    }
]


def chat(message: List[dict]) -> str:
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=message,
        tools=tools,
        temperature=0.7
    )
    print("==" * 10)
    print(f"[{resp}]")
    print("==" * 10)

    msg = resp.choices[0].message

    if msg.tool_calls:
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in msg.tool_calls
            ]
        })

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = tool_map[tc.function.name](**args)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False)
            })

            final_resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages
            )
            return final_resp.choices[0].message.content

    # print(resp)
    return msg.content


if __name__ == '__main__':
    for i in range(20):
        input_str = input("输入:")
        messages.append({"role": "user", "content": input_str})
        response_str = chat(messages)
        messages.append({"role": "assistant", "content": response_str})
        print(f"回复: {response_str}")
