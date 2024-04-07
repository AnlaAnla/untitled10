import os
import shutil

father_dir_path = r'C:\Code\ML\Image\card_cls\train_pokemon01_224\train'

for i, dir_name in enumerate(os.listdir(father_dir_path)):
    dir_path = os.path.join(father_dir_path, dir_name)

    new_dir_name = 'pokemon_' + str(i)
    new_dir_path = os.path.join(father_dir_path, new_dir_name)

    os.renames(dir_path, new_dir_path)
print('end')
