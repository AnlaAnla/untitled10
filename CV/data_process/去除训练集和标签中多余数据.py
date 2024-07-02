import os

train_path = r"C:\Code\ML\Image\yolo_data02\Card_2box08\train"
labels_path = r"C:\Code\ML\Image\yolo_data02\Card_2box08\labels"

# 去除没有label的图片

label_names = []
for label_name in os.listdir(labels_path):
    label_names.append(os.path.splitext(label_name)[0])
    print(label_name)

del_num = 0
for img_name in os.listdir(train_path):
    img_path = os.path.join(train_path, img_name)

    if os.path.splitext(img_name)[0] not in label_names:
        del_num += 1
        os.remove(img_path)
        print(f"{del_num} 删除: {img_name}")
print('end')
