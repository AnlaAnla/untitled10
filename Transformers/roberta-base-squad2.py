from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
QA_input = {
    'question': 'what is this year',
    'context': '''
    No. 98

2023-24 PANINI -PRIZM BASKETBALL
帕尼尼©2023 Panini America, Inc, ④ 2023 NBA Properties, Ine.
Produced in thie USA, ©2023 the Hationa/ Baskethall Players Association, All ights reserved.
    '''
}
res = nlp(QA_input)
print(res)
# b) Load model & tokenizer
# model = AutoModelForQuestionAnswering.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
