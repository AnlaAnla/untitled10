import os

father_dir = r"E:\College\college\image\[毛玉牛乳 (玉之けだま)] 汉化合集 V2"

i = 0
data = []
for dir_name in os.listdir(father_dir):
    dir_path = os.path.join(father_dir, dir_name)
    image_names = [name for name in os.listdir(dir_path) if
                   name.endswith('.jpg') or name.endswith('.png') or name.endswith('.jpeg')]
    data.append([i, dir_name, image_names])
    i += 1

print(data)