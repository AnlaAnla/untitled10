import os
import shutil

source_dir = r"D:\Code\ML\images\Mywork3\card_dataset"

total = 0
dir_nums = len(os.listdir(source_dir))

for name in os.listdir(source_dir):
    num = len(os.listdir(os.path.join(source_dir, name)))
    total += num

    print(name, ': ', num)

    print("{:<20}: {}".format(name, num))
    # if num < 12:
    #     print('-----------rm', name)
    #     shutil.rmtree(os.path.join(source_dir, name))

print("dir_nums: {}, tatal:{}, mean:{}".format(dir_nums, total, total // dir_nums))
