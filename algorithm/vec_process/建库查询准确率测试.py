import os

from MyOnnxModel_Resnet50 import MyOnnxModel
from MyOnnxYolo import MyOnnxYolo
import numpy as np


def vec_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def add_img2vector(img_path, img_name):
    global img_id, vec_data, name_list

    output = onnxModel.run(img_path)

    img_id += 1
    name_list.append(img_name)
    vec_data = np.concatenate([vec_data, output], axis=0)
    print('yes: ', img_id, img_name)


def search_img2vector(img_path: str):
    global img_id, vec_data, name_list

    output = onnxModel.run(img_path)

    distances = np.apply_along_axis(vec_distance, 1, vec_data, output)

    min_dis = np.min(distances)
    search_name = name_list[np.argmin(distances)]

    print(search_name, ': ', min_dis)
    return search_name


if __name__ == '__main__':
    onnxModel = MyOnnxModel(r"C:\Code\ML\Model\resent_out17355_AllCard08.onnx")
    # onnxYolo_card = MyOnnxYolo(r"C:\Code\ML\Model\onnx\yolo_card03.onnx")

    vec_data = np.zeros((1, 2048))
    name_list = ['background']

    img_id = 0

    data_dir = r'C:\Code\ML\Image\Card_test\mosic_prizm\prizm_yolo\base19-20 database'
    for img_dir_name in os.listdir(data_dir):
        for img_name in os.listdir(os.path.join(data_dir, img_dir_name)):
            img_path = os.path.join(data_dir, img_dir_name, img_name)
            add_img2vector(img_path, img_dir_name)
    print('end')

    # 进行准确率测试
    total_num = 0
    yes_num = 0
    val_dir = r"C:\Code\ML\Image\Card_test\mosic_prizm\prizm_yolo\base 19-20 val"
    for img_dir_name in os.listdir(val_dir):
        for img_name in os.listdir(os.path.join(val_dir, img_dir_name)):
            img_path = os.path.join(val_dir, img_dir_name, img_name)

            search_name = search_img2vector(img_path)

            total_num += 1

            if search_name == img_dir_name:
                yes_num += 1
            else:
                print('❌')
            print(total_num, '\t这张是:', img_dir_name, '\t查询: ', search_name)
            print('==' * 20)
            print()

    print('total_num:', total_num)
    print('yes_num:', yes_num)
    print(yes_num / total_num)
