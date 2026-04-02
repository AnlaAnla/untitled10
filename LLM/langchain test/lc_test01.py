import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# 0. 配置环境变量 (确保你的配置是对的)
os.environ["OPENAI_API_KEY"] = "sk-4395948fb2274a2fb7ef3ac4cc7b8804"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com"

# 1. 初始化模型
llm = ChatOpenAI(model="deepseek-chat", max_tokens=1024)

# 2. 严格定义下游任务需要的 JSON 结构
class UserIntent(BaseModel):
    intent_type: str = Field(description="用户意图，只能是 '查询', '闲聊', '投诉' 中的一个")
    entities: dict = Field(description="提取到的关键实体，例如 {'产品': '手机', '故障': '碎屏'}")
    urgency: int = Field(description="紧急程度，1-5分")

# 3. 将 Schema 绑定到模型
# 【修改点 1】：显式指定使用 json_mode
structured_llm = llm.with_structured_output(
    UserIntent,
    method="function_calling"
)

# 4. 构建新链
# 【修改点 2】：System prompt 中必须明确要求输出 JSON，这是 DeepSeek API 的硬性规定
prompt_extract = ChatPromptTemplate.from_messages([
    ("system", "你是一个精准的NLU意图识别模块。请根据用户的输入提取信息。"),
    ("user", "{text}")
])

extract_chain = prompt_extract | structured_llm

# 5. 测试中文复杂输入
test_text = "我昨天在你们这买的扫地机器人，今天一开机就冒烟了！你们怎么做品控的？立刻给我退款！"
print("正在请求 DeepSeek 解析意图，请稍候...")
result = extract_chain.invoke({"text": test_text})

# 打印出的直接是一个 Python Pydantic 对象
print(f"意图类型: {result.intent_type}") # 输出: 投诉
print(f"关键实体: {result.entities}")    # 输出: {'产品': '扫地机器人', '故障': '冒烟'}
print(f"紧急程度: {result.urgency}")     # 输出: 5