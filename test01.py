from llama_index.core import Document, SummaryIndex
from llama_index.core.node_parser import SentenceSplitter

# 假设 whisper_output 是 Whisper 识别后的结果列表
whisper_output = [
    {"start": 0.0, "end": 5.0, "text": "大家好，欢迎来到今天的拆卡直播！"},
    {"start": 5.2, "end": 10.0, "text": "今天我们要拆的是最新的 NBA Prizm 盒子。"},
    # ... 更多数据
]

# 1. 预处理数据：将时间戳格式化并拼接到文本前
# 格式化函数：将秒数转换为 MM:SS
def format_timestamp(seconds):
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{int(s):02d}"

formatted_text = ""
for segment in whisper_output:
    timestamp = format_timestamp(segment['start'])
    # 将时间戳硬编码进文本，这样 AI 总结时能直接看到
    formatted_text += f"[{timestamp}] {segment['text']}\n"

print(formatted_text)
# 2. 创建 LlamaIndex 文档对象
# 如果视频很长，建议切分成多个 Document，但在创建 Document 时保持作为一个整体传入
doc = Document(text=formatted_text)

# 3. 既然是做总结，不需要太碎的切片，但为了防止超长，还是用 Splitter 处理一下
# chunk_size 设置大一点，保证上下文连贯
splitter = SentenceSplitter(chunk_size=4096, chunk_overlap=200)
nodes = splitter.get_nodes_from_documents([doc])


print("="*20)
print(doc)
print(splitter)
print(nodes)


from llama_index.core import PromptTemplate

# 定义定制化的总结模板
# 这里的 {context_str} 是 LlamaIndex 会自动填入的视频文本
summary_template_str = """
# Role (角色设定)
你是一位资深的球星卡(NBA/Soccer)和宝可梦(TCG)拆卡专家。你的任务是根据一段英文直播的语音识别文本（带时间戳），为中国观众生成一份中文的“拆卡高光时刻总结”。

# Goal (目标)
用户听不懂英文，但想知道在视频的什么时间点开出了什么卡，特别是值钱的“好卡(Hits)”。你需要根据主播的**语气激动程度**和**专业术语**来推测卡片的稀有度。

# Workflow (工作流)
1. **分析文本**：阅读英文原文，忽略闲聊（如快递、天气），专注于开卡过程。
2. **提取信息**：识别球员名/宝可梦名、卡种（Base, Refractor, Auto等）、系列名。
3. **判断价值**：
   - **高价值信号 (High Value)**: 听到 "Auto", "Rookie (RC)", "Patch", "Jersey", "Serial Numbered" (如 /99, /10), "1 of 1", "SSP", "Gold", "Holo", "Alt Art", "Secret Rare"。
   - **情绪信号**: 听到 "Boom", "Bang", "Oh my god", "Let's go", "Huge hit", "Screaming"。
4. **中文输出**：将术语转换为中国卡迷通用的中文术语（见下文对照表），生成结构化列表。

# Terminology Mapping (术语对照参考)
- Auto / Autograph -> 签字
- Patch / Jersey -> 球衣切片 / 实物卡
- Rookie / RC -> 新秀卡
- Serial Numbered -> 带编 (如 /99)
- Parallel / Refractor -> 折射 / 平行
- One of One -> 1/1 (全球一张)
- Base -> 普卡 (一般不记录，除非是超级巨星)
- Holo / Shiny -> 闪卡

# Input Data (输入数据)
(请在此处粘贴你的 JSON 内容，如果内容太长可以分段发)

# Output Format (输出格式)
请输出一个 Markdown 表格，格式如下：

| 时间 (Time) | 视频部分 | 卡片名称 (中英对照) | 卡片描述/配置 | 稀有度/激动指数 |
|---|---|---|---|---|
| 05:23 | Part 1 | Lebron James (勒布朗·詹姆斯) | 湖人队 - 金折 /10 | 🔥🔥🔥🔥🔥 (超级大货) |
| 12:40 | Part 2 | Pikachu (皮卡丘) | 151系列 - 全图人物 (Full Art) | 🔥🔥🔥 (不错的一张) |

**要求：**
1. **时间转换**：将 JSON 中的秒数转换为 `MM:SS` 格式。
2. **稀有度分级**：
   - 🔥🔥🔥🔥🔥 (主播尖叫，1 of 1，顶级签字)
   - 🔥🔥🔥 (带编，普通签字，好卡)
   - 🔥 (普通折射，小特卡)
3. 如果该时间段没有开出卡片，不要输出。
4. **必须使用中文总结**，保留必要的英文原名（方便用户核对）。
"""

summary_prompt = PromptTemplate(summary_template_str)