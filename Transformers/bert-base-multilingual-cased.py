from transformers import pipeline

unmasker = pipeline('fill-mask', model='bert_result01-multilingual-cased')
unmasker("Hello I'm a [MASK] model.")
