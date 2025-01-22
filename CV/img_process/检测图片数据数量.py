import os
import shutil

source_dir = r"D:\Code\ML\Image\_CLASSIFY\card_cls2\train_serices_cls_data_yolo224\train"

total = 0
dir_nums = len(os.listdir(source_dir))

for name in os.listdir(source_dir):
    num = len(os.listdir(os.path.join(source_dir, name)))
    total += num

    print(name, ': ', num)

    if num == 0:
        print("               !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # if num < 12:
    #     print('-----------rm', name)
    #     shutil.rmtree(os.path.join(source_dir, name))

print("dir_nums: {}, tatal:{}, mean:{}".format(dir_nums, total, total // dir_nums))
