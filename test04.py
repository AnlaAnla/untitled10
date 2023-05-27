import os

path = r"D:\Download\Loli\国学\30"

for name in os.listdir(path):
    if name.count('1.txt') == 1:
        new_name = name.replace('1.txt', '')
        os.rename(os.path.join(path, name), os.path.join(path, new_name))

print('end')