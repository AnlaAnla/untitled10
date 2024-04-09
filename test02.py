import os

father_dir = r'C:\Code\ML\Image\card_cls\train_data6\train'

for dir_name in os.listdir(father_dir):
    dir_path = os.path.join(father_dir, dir_name)
    num = len(os.listdir(dir_path))

    if num < 1:
        print(dir_name, 'has less than 3 images -- ', num)
print('end')