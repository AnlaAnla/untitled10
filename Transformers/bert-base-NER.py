from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("dslim/distilbert-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/distilbert-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = '''
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

ner_results = nlp(example)
print(ner_results)
for data in ner_results:
    print(f"{data['entity']}: {data['word']}")

print()
