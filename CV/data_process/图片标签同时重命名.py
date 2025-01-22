import os
import uuid


# 图片和标签文件夹同时重命名
def rename_img_label(img_dir_path, labels_dir_path):
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


# 单个文件夹随机重命名
def random_rename_img(img_dir_path):
    for i, img_all_name in enumerate(os.listdir(img_dir_path)):
        img_name, img_suffix = os.path.splitext(img_all_name)
        img_path = os.path.join(img_dir_path, img_all_name)

        new_img_name = uuid.uuid4().hex + img_suffix
        new_img_path = os.path.join(img_dir_path, new_img_name)
        os.renames(img_path, new_img_path)

        print(img_all_name, " --> ", new_img_name)


if __name__ == '__main__':
    img_dir_path = r"D:\Code\ML\Image\_YOLO\yolo_data02\POKEMON reflect\0719POKEMON 折射样本3\train"
    labels_dir_path = r"D:\Code\ML\Image\_YOLO\yolo_data02\POKEMON reflect\0719POKEMON 折射样本3\labels"

    # rename_img_label(img_dir_path, labels_dir_path)

    random_rename_img(r"D:\Code\ML\Image\_YOLO\Yolo_card_seg\Card_1-16\images")
