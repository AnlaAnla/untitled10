from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
from threading import Thread
import time

prompt = '''
你是一个专业的体育卡牌数据解析助手。请严格按照以下规则, 从文本中提取卡牌信息，输出JSON格式结果：
**绝对不要包含任何其他文本、解释、问候语或对话。只输出 JSON 对象。**

1. 字段说明(输出字段必须为以下名称)：
   - year: 年份(取前四位数字，如'2023-24'取2023)
   - program: 卡系列(例如 Select/Honors/Playoffs/Momentum/Hoops/Leather and Lumber/Court Kings/Contenders Draft Picks/Prizm)
   - card_set: 卡种(卡系列后的第一个主要特征短语)
   - card_num: 卡编号(以#开头的最早出现的连续字符)
   - athlete: 球员名称(文本里出现的人名，需包含姓和名)

2. 处理规则：
   - 字段不存在时设为空字符串
   - 忽略括号内内容和特殊标记(如RC/SP)
   - 保持原始文本顺序，不要重组内容
   - 优先识别球员名称

3. 示例：
Bryan Bresee 2023 Donruss Optic #276 Purple Shock RC New Orleans Saints
{"year":"2023","program":"Donruss Optic","card_set":"Purple Shock","card_num":"276","athlete":"Bryan Bresee"}

请根据以上信息处理下面的文本
'''


class DeepSeekModel:
    def __init__(self, model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True  # Add trust_remote_code here as well
        )
        self.model = torch.compile(self.model)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Get device after model loading
        print(f"Using device: {self.device}")

    def generate_response(self, user_input, max_new_tokens=1024, temperature=0.6, stream=False):

        inputs = self.tokenizer([user_input], return_tensors="pt").to(self.device)
        prompt_length = len(inputs["input_ids"][0])

        if stream:
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            return streamer  # Return the streamer directly

        else:
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                #  从生成结果中去除 prompt 部分
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return response
            except torch.cuda.OutOfMemoryError:
                print("显存不足，请尝试减小 max_new_tokens 或 batch size。")
                return "Error: Out of Memory"
            except Exception as e:
                print(f"生成回答时发生错误: {str(e)}")
                return f"Error: {str(e)}"


def process_question(model, question, stream=False):
    if stream:
        print("\nDeepSeek: ", end="")
        for response in model.generate_response(question, stream=True):
            print(response, end="", flush=True)
        print()  # Newline after streaming
    else:
        response = model.generate_response(question)
        print("\nDeepSeek:", response)


def main():
    model = DeepSeekModel()

    while True:
        user_input = input("\n请输入您的问题 (输入 'quit' 退出): ")
        if user_input.lower() == 'quit':
            break

        stream_choice = input("是否使用流式输出？(y/n): ")
        process_question(model, user_input, stream=stream_choice.lower() == 'y')


def get_result(model, question):
    t1 = time.time()
    full_prompt = f"{prompt}\n{question}\n<think>\n首先，我要拆分 [{question}] 这个字符串，找出每个字段的信息。然后分别提取 “year”、“program”、“card_set”、“card_num”、“athlete”字段 \n优先关注athlete字段,各个字段不存在时设为空字符串，所以如果某个字段在文本中没有出现，就留空, 不要输出重复内容\n同时要注意每个字段一般都不一样\n</think>\n"
    response = model.generate_response(full_prompt, max_new_tokens=128, temperature=0.1)
    print(response)
    print('time: ', time.time() - t1, ' s')


if __name__ == "__main__":
    model = DeepSeekModel()  # Load model only once

    question = "2021-22 Panini Select Nikola Jokic Fast Break Disco Prizm Courtside #211 Nuggets"

    get_result(model, question)
    print()
    # main()
