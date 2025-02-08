import pandas as pd
import os

check_list_dir = r"D:\Code\ML\Text\card"

program_list = []
for filename in os.listdir(check_list_dir):
    file_path = os.path.join(check_list_dir, filename)
    data = pd.read_csv(file_path, low_memory=False)
    data_list = list(data['card_set'].dropna().unique())

    print(f"{filename}: {len(data_list)}")
    program_list += data_list

print('merge data: ', len(program_list))
program_list = set(program_list)
print('merge set: ', len(program_list))



with open(r'D:\Code\ML\Text\checklist_tags\cardSet_tags.txt', 'w', encoding='utf-8') as f:
    for text in program_list:
        f.write(text + '\n')
print('end')
