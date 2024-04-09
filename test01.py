import os
import shutil

father_dir = r'C:\Code\ML\Image\Card_test\mosic_prizm\prizm\base 19-20 val'
save_father_dir = r'C:\Code\ML\Image\Card_test\mosic_prizm\prizm\base19-20 database'


for dir_name in os.listdir(father_dir):
    for img_name in os.listdir(os.path.join(father_dir, dir_name))[:1]:
        img_path = os.path.join(father_dir, dir_name, img_name)

        save_dir = os.path.join(save_father_dir, dir_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        img_new_path = os.path.join(save_dir, img_name)
        shutil.move(img_path, img_new_path)
        print(img_name)


print('end')