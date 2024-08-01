import os
import uuid

img_dir_path = r"C:\Code\ML\Image\yolo_data02\Card_scratch\0729pokemon\images"
labels_dir_path = r"C:\Code\ML\Image\yolo_data02\Card_scratch\0729pokemon\labels"

for i, img_all_name in enumerate(os.listdir(img_dir_path)):
    img_name = os.path.splitext(img_all_name)[0]
    img_path = os.path.join(img_dir_path, img_all_name)

    label_name = img_name + ".json"
    label_path = os.path.join(labels_dir_path, label_name)

    # print('have image path:', os.path.exists(img_path))
    # print('have label path:', os.path.exists(label_path))
    # print('_'*30)

    new_img_name = img_name + uuid.uuid4().hex
    new_img_all_name = new_img_name + ".jpg"
    new_img_path = os.path.join(img_dir_path, new_img_all_name)

    new_label_name = new_img_name + ".json"
    new_label_path = os.path.join(labels_dir_path, new_label_name)

    os.renames(img_path, new_img_path)
    if os.path.exists(label_path):
        os.renames(label_path, new_label_path)
    else:
        print("不存在label: ", label_path)
    print(i, ' ', new_img_name)
