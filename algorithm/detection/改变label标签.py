import os
import glob

label_dir = r"C:\Code\ML\Image\yolov8_data\hand.v1i.yolov8\valid\labels"

label_path_list = glob.glob(os.path.join(label_dir, "*"))

print(len(label_path_list))
print()

for i, label_path in enumerate(label_path_list):
    with open(label_path, "r", encoding='utf-8') as f:
        data = f.read()
    data = '2' + data[1:]
    with open(label_path, "w", encoding='utf-8') as f:
        f.write(data)
    print(i, label_path)