from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
import torch

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", device_map=device).eval()

    # tokenizer.save_pretrained('nllb-200-distilled-600M-ct2')
    # model.save_pretrained('nllb-200-distilled-600M')

    sentence_list = ['In fact,Pandas spend upwards of 12 to 16 hours per day gnawing...',
                     "he main entrypoint in Python is the Translator class which provides methods to translate files or batches as well as methods to score existing translations.",
                     "which takes a lot of work to chew.",
                     "Horses,Pandas,Hippos, orHummingbirds"]

    for text_to_translate in sentence_list:
        # text_to_translate = "今天是个好日子, 红红火火的好日子"
        model_inputs = tokenizer(text_to_translate, return_tensors="pt").to(device)

        t1 = time.time()
        # translate to eng_Latn, zho_Hans
        gen_tokens = model.generate(**model_inputs,
                                    forced_bos_token_id=tokenizer.convert_tokens_to_ids("zho_Hans"))
        output = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]
        print(output, time.time() - t1)
