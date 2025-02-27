import json
from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

text_prompt = """You are a professional sports card data parsing assistant. Please strictly follow the rules below to extract card information from text, and output the result in JSON format:

**Do not include any other text, explanations, greetings, or dialogue. Only output the JSON object.**

1. Field descriptions (output fields must use the following names):
   - year: Year (take the first four digits, e.g., '2023-24' becomes 2023)
   - program: Card series (e.g., Select/Honors/Playoffs/Momentum/Hoops/Leather and Lumber/Court Kings/Contenders Draft Picks/Cooperstown/Gala/Complete/Vertex/Innovation/Luminance/PhotoGenic/Diamond Kings/Prizm)
   - card_set: Card type (the first main feature phrase after the card series)
   - card_num: Card number (the earliest consecutive characters starting with #)
   - athlete: Player name (the last appearing full name)

2. Processing rules:
   - Set the field to an empty string if it does not exist
   - Ignore content in parentheses and special marks (e.g., RC/SP)
   - Maintain the original text order; do not reorganize the content
   - Prioritize matching the card series list; leave blank if not matched
   - Note that "Panini" does not belong to the program

3. For example:
Bryan Bresee 2023 Donruss Optic #276 Purple Shock RC New Orleans Saints
{"year":"2023","program":"Donruss Optic","card_set":"Purple Shock","card_num":"276","athlete":"Bryan Bresee"}

"""


# 定义数据分词函数
def tokenize_function(tokenizer, examples):
    inputs = [
        f"<s>[INST] {text_prompt}{item['input']} [/INST]"
        for item in examples["data"]]
    outputs = [item["output"] for item in examples["data"]]

    # 分词输入
    tokenized_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    # 分词输出
    tokenized_outputs = tokenizer(outputs, padding="max_length", truncation=True, max_length=256, return_tensors="pt")

    # 设置 labels 为输出的 input_ids
    tokenized_inputs["labels"] = tokenized_outputs["input_ids"]
    # 将注意力掩码中的 padding 位置设置为 -100
    for i, mask in enumerate(tokenized_outputs["attention_mask"]):
        for j in range(len(mask)):
            if mask[j] == 0:
                tokenized_inputs["labels"][i][j] = -100

    return tokenized_inputs


if __name__ == '__main__':
    # 数据
    data = [
        {"input": "2023-24 Spectra Brandon Miller Spectacular Debut Rookie Card RC  Base #193 ",
         "output": json.dumps(
             {"year": "2023", "program": "Spectra", "card_set": "Base Spectacular Debut", "card_num": "193",
              "athlete": "Brandon Miller"})},
        {"input": "2023-24 Panini Select LeBron James Silver Prizm",
         "output": json.dumps(
             {"year": "2023", "program": "Select", "card_set": "Silver Prizm", "card_num": "",
              "athlete": "LeBron James"})},
        {"input": "2023-24 Panini Mosaic Luka Doncic Bank Shot #16 Dallas Mavericks",
         "output": json.dumps({"year": "2023", "program": "Mosaic", "card_set": "Bank Shot", "card_num": "16",
                               "athlete": "Luka Doncic"})},
        {"input": "2023-24 Panini NBA Hoops - Hoops Tribute Winter #298 Victor Wembanyama (RC)",
         "output": json.dumps(
             {"year": "2023", "program": "Hoops", "card_set": "Base Hoops Tribute Winter", "card_num": "298",
              "athlete": "Victor Wembanyama"})},
        {"input": "2023-24  Select Concours Light Blue Disco Prizm Noah Clowney RC 45/99 Szs",
         "output": json.dumps(
             {"year": "2023", "program": "Select", "card_set": "Light Blue Disco Prizm", "card_num": "",
              "athlete": "Noah Clowney"})},
        {"input": "2023-24 Haunted Hoops #282 Victor Wembanyama Rising Stars Orange Rookie RC",
         "output": json.dumps(
             {"year": "2023", "program": "Hoops", "card_set": "Rising Stars Orange", "card_num": "282",
              "athlete": "Victor Wembanyama"})},
        {"input": "2023-24 Panini Mosaic Victor Wembanyama RC #12 Epic Performers Rookie Spurs",
         "output": json.dumps(
             {"year": "2023", "program": "Mosaic", "card_set": "Epic Performers Mosaic", "card_num": "12",
              "athlete": "Victor Wembanyama"})},
        {"input": "2023-24  Select Concours Light Blue Disco Prizm Colby Jones 84/99 RC Szs",
         "output": json.dumps(
             {"year": "2023", "program": "Select", "card_set": "Base Blue Disco Prizm", "card_num": "",
              "athlete": "Colby Jones"})},
        {"input": "2023-24 Panini Mosaic City Edition Silver Prizm #278 LeBron James LA Lakers",
         "output": json.dumps(
             {"year": "2023", "program": "Mosaic", "card_set": "City Edition Silver Prizm", "card_num": "278",
              "athlete": "LeBron James"})},
        {"input": "2023-24 Prizm Monopoly Stephen Curry SILVER Prizm Card #28 Warriors Star!",
         "output": json.dumps(
             {"year": "2023", "program": "Prizm", "card_set": "Base Prizms Silver", "card_num": "28",
              "athlete": "Stephen Curry"})},
    ]

    # 创建 Dataset
    dataset = Dataset.from_dict({"data": data})

    # 加载模型和分词器
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 准备模型
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # 训练参数
    training_args = TrainingArguments(
        num_train_epochs=25,  # 减少轮数
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,  # 降低学习率
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        report_to="none",
        logging_strategy="epoch",
    )

    # 分词数据集
    tokenized_dataset = dataset.map(lambda x: tokenize_function(tokenizer, x), batched=True, remove_columns=["data"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 创建 Trainer 并训练
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset, tokenizer=tokenizer,
                      data_collator=data_collator)
    trainer.train()

    # 保存模型
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
