from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0',
                                  trust_remote_code=True,
                                  low_cpu_mem_usage=True,
                                  device_map='cuda',
                                  use_safetensors=True,
                                  pad_token_id=tokenizer.eos_token_id)
model = model.eval().cuda()


def ocr(img_path, ocr_type='ocr'):
    res = model.chat(tokenizer, img_path, ocr_type=ocr_type)
    print(res)

# input your test image
image_file = r"D:\Code\ML\Image\_TEST_DATA\Card_series_cls\2023-24\2023-24 CONTENDERS\17-2023-24 PANINI CONTENDERS FIRST ROUND TICKET.jpg"

ocr(image_file)
print()

# plain texts OCR
# res = model.chat(tokenizer, image_file, ocr_type='ocr')

# format texts OCR:
# res = model.chat(tokenizer, image_file, ocr_type='format')

# fine-grained OCR:
# res = model.chat(tokenizer, image_file, ocr_type='ocr', ocr_box='')
# res = model.chat(tokenizer, image_file, ocr_type='format', ocr_box='')
# res = model.chat(tokenizer, image_file, ocr_type='ocr', ocr_color='')
# res = model.chat(tokenizer, image_file, ocr_type='format', ocr_color='')

# multi-crop OCR:
# res = model.chat_crop(tokenizer, image_file, ocr_type='ocr')
# res = model.chat_crop(tokenizer, image_file, ocr_type='format')

# render the formatted OCR results:
# res = model.chat(tokenizer, image_file, ocr_type='format', render=True, save_render_file = './demo.html')

# print(res)




