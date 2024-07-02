from transformers import pipeline

unmasker = pipeline('fill-mask', model='bert-base-multilingual-cased')
unmasker("Hello I'm a [MASK] model.")
