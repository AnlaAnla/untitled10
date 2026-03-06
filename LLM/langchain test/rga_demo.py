from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# === 1. 准备大模型 (vLLM) ===
llm = ChatOpenAI(
    openai_api_base="http://192.168.77.249:7676/v1",
    openai_api_key="EMPTY",
    model_name="Qwen/Qwen2.5-1.5B-Instruct"
)

# === 2. 准备向量搜索 (你的主场) ===
# 假设我们用一个开源的中文词向量模型 (BGE模型)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5",
                                   model_kwargs={'device': 'cpu'})

# 模拟一份我们公司的“私有知识库”文本
texts = [
    "我们公司的新产品叫『AI-Star』，发布于2026年。",
    "FastAPI 是一个现代、快速（高性能）的 Web 框架，用于构建 APIs，基于 Python 3.8 类型提示。",
    "vLLM 采用了 PagedAttention 技术，极大地优化了显存中 KV Cache 的管理。"
    "散兵是原神里面的一个角色, 它是出生于雷神, 现在在须弥国"
]

# 直接利用文本和 Embedding 模型构建一个存在内存里的向量库（FAISS）
vectorstore = FAISS.from_texts(texts, embedding=embeddings)

# 将向量库转化成一个“检索器 (Retriever)”
# 相当于封装了你的：搜索Top-K向量 -> 提取对应文本 的逻辑
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# === 3. 构建 RAG Prompt 模板 ===
template = """
请你根据下面的上下文内容，回答用户的问题。如果上下文中没有包含相关信息，请直接回答"我不知道"。

上下文内容：
{context}

用户问题：{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# === 4. 组装 RAG 链 (LangChain 的黑魔法) ===
# 这里的逻辑是：
# 1. 把用户的 question 传给 retriever 找出 context
# 2. 把用户的 question 原样向后传 (RunnablePassthrough)
# 3. 填入 prompt，发给 llm，最后输出字符串
rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

# === 5. 测试效果 ===
# print("--- 测试 1：询问私有知识 ---")
# res1 = rag_chain.invoke("我们公司的新产品叫什么？什么时候发布的？")
# print(res1)
#
# print("\n--- 测试 2：询问专业知识 ---")
# res2 = rag_chain.invoke("vLLM 是如何优化显存的？")
# print(res2)

print("\n--- 测试 3：超纲问题 (防幻觉) ---")
res3 = rag_chain.invoke("昨晚的国足比分是多少？")
print(res3)

print("\n--- 测试 3：超纲问题 (防幻觉) ---")
res3 = rag_chain.invoke("散兵现在在哪里?")
print(res3)
