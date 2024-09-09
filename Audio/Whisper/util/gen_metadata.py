import os
import pandas as pd
import json
import numpy as np

label_dir_path = r"D:\Code\ML\Audio\Data1\info"

data = []

for label_name in os.listdir(label_dir_path):
    label_path = os.path.join(label_dir_path, label_name)

    with open(label_path, 'r', encoding='utf-8') as f:
        label_json = json.load(f)

    path = label_json['audio']['path']
    sentence = label_json['sentence']

    data.append([path, sentence])

print(data)
data = np.array(data)
data = pd.DataFrame(data, columns=['file_name', 'sentence'])
print(data)
data.to_csv(r"D:\Code\ML\Audio\Data1\metadata.csv", encoding='utf-8', index=False)
print('end')