import os
import glob
import shutil

source_dir = r"D:\Code\ML\images\Mywork3\card_dataset_yolo\train"
dst_dir = r"D:\Code\ML\images\Mywork3\card_dataset_yolo\val"

for i, dir_name in enumerate(os.listdir(source_dir)):

    img_dir = os.path.join(source_dir, dir_name)
    img_name = os.listdir(img_dir)[0]
    img_path = os.path.join(img_dir, img_name)

    save_dir = os.path.join(dst_dir, dir_name)
    save_path = os.path.join(save_dir, img_name)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print("{:<80} ==>> {}".format(img_path, save_path))
    shutil.copy(img_path, save_path)
    print(i)

print('end')