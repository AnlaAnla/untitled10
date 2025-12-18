from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

question_list = ["What is this person name?",
                 "What is this team?",
                 "What is the year information near symbol 'base'?"]
context = '''
No. 196
DEVONTE'
GRAHAM
Rested, rejuvenated and reloaded, Graham came out firing after the 2020-21 AiI-Star break. He shot 46.8 percent from 3-point range and 94.7 percent from the free-throw line over his next 10 games from March 11-28, averaging 14.3 points, 3.8 assists and 3.7 made 3-pointers per game as the Hornets compiled a 6-4 record.
SEASON
1920
TEAM HORNETS NBA TOTALS
G FG% FT% 3PM RPG APG 63.382820 218 3.4 7.5 109.375.810 252 2.5
5.4
STL BLK PTS PPG
62
85
15 1145 18.2
1362 12.5
2020-21 PANINI-MOSAIC BASKETBALL帕尼尼©2021 Panini America,Inc.© 2021 NBA Properties, Inc. Produced in the USA.© 2021 Officially licensed product of the National Baskethall Players Association.
    '''

for question in question_list:
    QA_input = {
        'question': question,
        'context': context
    }
    res = nlp(QA_input)
    print(res)
