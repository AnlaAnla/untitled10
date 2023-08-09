import os
import shutil

'''
---source_dir
--sub_dir1
-img1
-img2
...
--sub_dir2
-img1
-img2
...
'''
# 遍历训练集的文件夹，判断文件夹数，
# is_delete: 判断子文件夹图片数量是否<=threshold_num, 是则删除子文件夹
def check_num(source_dir, is_delete=False, threshold_num = 100):
    total = 0
    dir_nums = len(os.listdir(source_dir))

    for name in os.listdir(source_dir):
        num = len(os.listdir(os.path.join(source_dir, name)))
        total += num

        print(name, ': ', num)

        if is_delete:
            print("{:<20}: {}".format(name, num))
            if num <= threshold_num:
                print('-----------rm', name)
                shutil.rmtree(os.path.join(source_dir, name))

    print("dir_nums: {}, tatal:{}, mean:{}".format(dir_nums, total, total // dir_nums))

if __name__ == '__main__':
    source_dir = r"D:\Code\ML\images\Mywork3\card_dataset"
    check_num(source_dir)