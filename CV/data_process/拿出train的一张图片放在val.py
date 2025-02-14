import os
import glob
import shutil


# 遍历source_dir中的所有子目录，并从每个子目录中复制第n个文件到dst_dir中的相应子目录
# 如果目标子目录不存在，会创建一个。
source_dir = r"D:\Code\ML\Image\_CLASSIFY\card_cls2\train_serices_cls_data_yolo224\train"
dst_dir = r"D:\Code\ML\Image\_CLASSIFY\card_cls2\train_serices_cls_data_yolo224\val"
num_of_move = 4


if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)

for i, dir_name in enumerate(os.listdir(source_dir)):

    img_dir = os.path.join(source_dir, dir_name)

    dir_img_num = len(os.listdir(img_dir))
    remove_num = num_of_move if num_of_move < dir_img_num else dir_img_num
    print(remove_num)
    img_names = sorted(os.listdir(img_dir))[:num_of_move]

    save_dir = os.path.join(dst_dir, dir_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        save_path = os.path.join(save_dir, img_name)

        print("{:<80} ==>> {}".format(img_path, save_path))
        # 如果生成测试集，使用move，生成验证集使用copy
        # shutil.move(img_path, save_path)
        shutil.copy(img_path, save_path)
    print(i)

print('end')
