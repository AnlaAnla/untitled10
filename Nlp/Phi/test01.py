import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}


messages = [
    {"role": "system", "content": "你是一个语气活泼的机器人助手"},
]

while True:
    you_input = input(">>>")
    # messages.append({"role": "user", "content": you_input})
    messages = [
        {"role": "system", "content": "你扮演一个智慧的旅行家, 你走过地球各种地方, 喜欢讲解世界上的各种历史文化和故事"},
        {"role": "user", "content": you_input}
    ]

    output = pipe(messages, **generation_args)
    response = output[0]['generated_text']

    print(response)
    # messages.append({"role": "system", "content": response})
    print('=============')

