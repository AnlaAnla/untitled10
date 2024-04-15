import json
import requests

with open('aa.json', 'r', encoding='utf-8') as f:
    data = f.read()

url = "http://localhost:8000/train/params_json"
response = requests.post(url, files={"file": data})

print(response)