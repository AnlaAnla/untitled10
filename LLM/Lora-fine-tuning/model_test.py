import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import time

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

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
model = PeftModel.from_pretrained(model, "./fine_tuned_model")
model = torch.compile(model)

def precess(ebay_text):
    t1 = time.time()
    input_text = f"<s>[INST] {text_prompt}{ebay_text} [/INST]"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    # 生成输出
    outputs = model.generate(**inputs,
                             max_new_tokens=96,
                             num_beams=5,
                             early_stopping=True)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(result.replace(f"[INST] {text_prompt}{ebay_text} [/INST]", '').strip())
    print(f"--- {time.time() - t1:.4f} seconds ---")



# 测试
precess("Trayce Jackson-Davis 2023-24 Panini Mosaic Rookie Red Yellow Fusion /75 RC #205")
precess("2023-24 Panini Prizm - Red White and Blue Prizm #150 Amen Thompson (RC)")
precess("2023-24 Caitlin Clark Collection Raining 3s #R1 + #R2 Iowa Hawkeyes RC Fever")
precess("2023-24 Prizm Monopoly Stephen Curry SILVER Prizm Card #28 Warriors Star!")
print()
