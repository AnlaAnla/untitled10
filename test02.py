import os

dir_path = r"C:\Code\ML\Image\test02"


for i, filename in enumerate(os.listdir(dir_path)):
    img_path = os.path.join(dir_path, filename)
    new_path = os.path.join(dir_path, 'a' + str(i) + ".jpg")
    os.rename(img_path, new_path)