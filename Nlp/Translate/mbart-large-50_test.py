from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import time

if __name__ == '__main__':
    device = "cuda"
    src_lang = "en_XX"
    tgt_lang = "zh_CN"

    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", device_map=device).eval()
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    # tokenizer.save_pretrained('mbart-large-50-many-to-many-mmt-ct2')
    # model.save_pretrained('mbart-large-50-many-to-many-mmt')

    sentence_list = ['I have a 长颈鹿, 今天天气不错,In fact,Pandas spend upwards of 12 to 16 hours per day gnawing...',
                     "he main entrypoint in Python is the Translator class which provides methods to translate files or batches as well as methods to score existing translations.",
                     "which takes a lot of work to chew.",
                     "that is Horses,Pandas,Hippos, or Hummingbirds"]

    for text_to_translate in sentence_list:
        tokenizer.src_lang = src_lang
        model_inputs = tokenizer(text_to_translate, return_tensors="pt", max_length=512, truncation=True).to(device)

        t1 = time.time()
        generated_tokens = model.generate(
            **model_inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
        )
        output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        print(output, time.time() - t1)
