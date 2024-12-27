import ctranslate2
import transformers
import os
import time
import torch
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def translate(text, tgt_lang):
    source = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    target_prefix = [tgt_lang]
    results = translator.translate_batch([source], target_prefix=[target_prefix])
    target = results[0].hypotheses[0][1:]

    return tokenizer.decode(tokenizer.convert_tokens_to_ids(target), skip_special_tokens=True)


if __name__ == '__main__':
    device = "cuda"

    src_lang = "en_XX"
    tgt_lang = "zh_CN"

    model_path = r"mbart-large-50-many-to-many-mmt-ct2"

    translator = ctranslate2.Translator(model_path, device=device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, src_lang=src_lang)

    sentence_list = ['I have a 长颈鹿, 今天天气不错,In fact,Pandas spend upwards of 12 to 16 hours per day gnawing...',
                     "he main entrypoint in Python is the Translator class which provides methods to translate files or batches as well as methods to score existing translations.",
                     "which takes a lot of work to chew.",
                     "that is Horses,Pandas,Hippos, or Hummingbirds"]

    for text in sentence_list:
        t1 = time.time()
        print(translate(text, tgt_lang), time.time() - t1)
