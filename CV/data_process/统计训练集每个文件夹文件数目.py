import os

father_dir = r"D:\Code\ML\Image\_CLASSIFY\card_cls2\Serices_cls_data_yolo224\train"

total_num = 0
dir_nums = len(os.listdir(father_dir))
max_dir_num = 0
min_dir_num = 99999999

for sub_dir_name in os.listdir(father_dir):
    sub_dir_path = os.path.join(father_dir, sub_dir_name)

    dir_img_num = len(os.listdir(sub_dir_path))
    if dir_img_num > max_dir_num:
        max_dir_num = dir_img_num
    if dir_img_num < min_dir_num:
        min_dir_num = dir_img_num
    total_num += dir_img_num

    print(f"{sub_dir_name:<30}: {dir_img_num}")
    # for img_name in os.listdir(sub_dir_path):
    #     img_path = os.path.join(sub_dir_path, img_name)
    #     if not img_name.endswith(".jpg"):
    #         print(img_path)
    #         os.remove(img_path)
print(f'\n共{total_num}, 有{dir_nums}个文件夹, 均值为:{total_num/dir_nums}, max: {max_dir_num}, min: {min_dir_num}')
