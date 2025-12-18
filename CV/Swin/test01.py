import os

data_path =r"/home/martin/ML/Image/CardCls/panini_archive_resize384/train"

print(len(os.listdir(data_path)))

card_dict = {}
for i, dir_name in enumerate(os.listdir(data_path)):
    dir_path = os.path.join(data_path, dir_name)

    tag_list = dir_name.split(", ")
    new_tag = f"{tag_list[0]}, {tag_list[1]}, {tag_list[4]}, {tag_list[5]}"
    # print(new_tag)
    if new_tag not in card_dict:
        card_dict[new_tag] = 1
    else:
        card_dict[new_tag] += 1

print(card_dict)
print(len(card_dict))
print("mean", sum(card_dict.values())/len(card_dict))
