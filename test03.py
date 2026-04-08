import os

dir_path = r"C:\Code\ML\Image\_CLASSIFY\card_cls2\Pokemon01\pokemon_cn"

count = 0
name_list = []
for file_name in os.listdir(dir_path):
    names = file_name.split(",")
    if len(names) == 3:
        item = f"{names[1]},{names[2]}"
    if len(names) == 4:
        item = f"{names[1]},{names[2]},{names[3]}"
    # print(item)
    if item in name_list:
        count += 1
        print(file_name)
        # break
    name_list.append(item)

print(count)
