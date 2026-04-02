import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# 0. 配置环境变量 (确保你的配置是对的)
os.environ["OPENAI_API_KEY"] = "sk-4395948fb2274a2fb7ef3ac4cc7b8804"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com"

# 1. 初始化模型
llm = ChatOpenAI(model="deepseek-chat", max_tokens=1024)


# 1. 定义算法/业务工具 (使用 @tool 装饰器)
@tool
def check_order_status(order_id: str) -> str:   
    """查询电商订单的实时状态，必须传入订单号。"""
    # 模拟查库逻辑
    mock_db = {"D1001": "已发货", "D1002": "正在退款处理中"}
    return mock_db.get(order_id, "未找到订单记录")


@tool
def calculate_refund_amount(order_id: str) -> float:
    """计算订单退款金额。"""
    return 199.50 if order_id == "D1002" else 0.0


tools = [check_order_status, calculate_refund_amount]

# 1. 直接使用 prebuilt 创建一个标准的 ReAct 智能体循环图
# 引擎逻辑：接收输入 -> LLM思考 -> 需要调工具吗？
# -> 是 -> 执行 Python 函数 -> 将结果写入历史 -> 再次丢给 LLM思考
# -> 否 -> 直接输出最终答案 -> 结束
agent = create_react_agent(llm, tools=tools)

# 2. 执行 Agent 图
# 我们开启 stream 流式输出，观察它的内部运行轨迹
messages = [HumanMessage(content="帮我查一下订单 D1002 的状态，另外我能收到多少退款？")]

print("=== 智能体运行轨迹 ===")
for event in agent.stream({"messages": messages}):
    # event 的 key 会是运行时的节点名称 (如 'agent' 代表模型节点，'tools' 代表工具节点)
    for node_name, node_state in event.items():
        print(f"\n[执行节点]: {node_name}")
        # 打印该节点产生的最新消息
        latest_msg = node_state["messages"][-1]

        if node_name == "agent":
            if latest_msg.tool_calls:
                print(f"模型决定调用工具: {[t['name'] for t in latest_msg.tool_calls]}")
            else:
                print(f"模型最终回答: {latest_msg.content}")

        elif node_name == "tools":
            print(f"工具执行完毕，返回结果: {latest_msg.content}")
