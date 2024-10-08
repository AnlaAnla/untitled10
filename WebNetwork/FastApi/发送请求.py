import json
import requests

with open('train_params.json', 'r', encoding='utf-8') as f:
    data = f.read()
url = "http://192.168.56.116:6666/train/params_json"
response = requests.post(url, files={"file": data})

print(response)
