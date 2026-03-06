from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# === 1. 连接本地 vLLM 大模型 ===
llm = ChatOpenAI(
    openai_api_base="http://192.168.77.249:7676/v1",
    openai_api_key="EMPTY",
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    temperature=0.1,
    max_retries=2
)

# === 2. 准备工具 ===
# 实例化搜索引擎工具
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

# === 3. 创建现代版 Agent (使用 LangGraph) ===
# 这一行代码替代了老版本复杂的 prompt、agent、executor 组装
agent_executor = create_react_agent(llm, tools)

# === 4. 测试运行 ===
print("\n========== 开始测试 ==========\n")


def run_agent(question):
    print(f"👤 用户问题: {question}")

    # LangGraph 使用 "messages" 列表来管理对话状态
    # stream_mode="values" 可以让我们看到思考过程
    events = agent_executor.stream(
        {"messages": [HumanMessage(content=question)]},
        stream_mode="values"
    )

    for event in events:
        # 打印出模型每次生成的最新消息（思考过程或最终结果）
        latest_message = event["messages"][-1]

        # 如果是 AI 发出的消息
        if latest_message.type == "ai":
            if latest_message.tool_calls:
                print(f"🤖 [思考中] 决定使用工具: {latest_message.tool_calls[0]['name']}")
                print(f"   传入参数: {latest_message.tool_calls[0]['args']}")
            elif latest_message.content:
                print(f"🤖 [最终回答]: {latest_message.content}")

        # 如果是工具返回的消息
        elif latest_message.type == "tool":
            print(f"🛠️ [工具返回了搜索结果] (已隐藏过长内容...)")

    print("-" * 50)


# 测试 1：需要联网的问题
run_agent("目前最强的AI模型是哪几个?")
