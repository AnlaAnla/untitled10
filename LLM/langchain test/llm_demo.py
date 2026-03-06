from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 实例化 LLM（重点：把地址指向你本地的 vLLM 端口，伪装成 OpenAI）
llm = ChatOpenAI(
    openai_api_base="http://192.168.77.249:7676/v1", # vLLM 的服务地址
    openai_api_key="EMPTY",                     # 本地服务不需要真实的秘钥
    model_name="Qwen/Qwen2.5-1.5B-Instruct",    # 必须和 vLLM 启动时的名字一致
    temperature=0.7
)

# 2. 构建 Prompt 模板 (系统提示词 + 用户提问)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个资深的 Python 和算法工程师，回答要简洁专业。"),
    ("user", "{question}")
])

# 3. 核心概念：LCEL (LangChain Expression Language) 链式调用
# 把 Prompt、大模型、输出解析器 像管道一样连起来
chain = prompt | llm | StrOutputParser()

# 4. 执行调用
question = "请用一句话解释一下什么是 FastAPI？"
print(f"User: {question}")
response = chain.invoke({"question": question})
print(f"AI: {response}")
