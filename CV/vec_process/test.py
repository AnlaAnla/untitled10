import os

from MyOnnxModel_MobileNetV3 import MyOnnxModel
from MyOnnxYolo import MyOnnxYolo
import numpy as np


def vec_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def add_img2vector(img, img_name):
    global img_id, vec_data, name_list

    onnxYolo_card.set_result(img)
    input_array = onnxYolo_card.get_max_img(cls_id=0)

    output = onnxModel.run(input_array)

    distances = np.apply_along_axis(vec_distance, 1, vec_data, output)
    # 向量距离对比,判断是否重复
    min_dis = np.min(distances)

    print('_' * 20)
    print(min_dis)
    if min_dis > 7:
        img_id += 1
        name_list.append(img_name)
        vec_data = np.concatenate([vec_data, output], axis=0)

        print('yes: ', img_id, img_name)
        return False
    print("No!: ", img_id, '\t重复id:', name_list[np.argmin(distances)])
    return True


def search_img2vector(img):
    global img_id, vec_data, name_list
    onnxYolo_card.set_result(img)
    input_array = onnxYolo_card.get_max_img(cls_id=0)

    output = onnxModel.run(input_array)

    distances = np.apply_along_axis(vec_distance, 1, vec_data, output)

    min_dis = np.min(distances)
    search_name = name_list[np.argmin(distances)]

    print(search_name, ': ', min_dis)
    return search_name


if __name__ == '__main__':
    onnxModel = MyOnnxModel(r"C:\Code\ML\Model\onnx\mobilenetv3_features_AllCard09.onnx")
    onnxYolo_card = MyOnnxYolo(r"C:\Code\ML\Model\onnx\yolo_handcard01.onnx")

    vec_data = np.zeros((1, 960))
    name_list = ['background']

    img_id = 0

    data_dir = r'C:\Code\ML\Image\Card_test\OBJDetected\dataset poke-op'
    for img_dir_name in os.listdir(data_dir):
        for img_name in os.listdir(os.path.join(data_dir, img_dir_name)):
            img_path = os.path.join(data_dir, img_dir_name, img_name)
            add_img2vector(img_path, img_dir_name)
    print('end')

    total_num = 0
    yes_num = 0
    val_dir = r"C:\Code\ML\Image\Card_test\OBJDetected\val"
    for img_dir_name in os.listdir(val_dir):
        for img_name in os.listdir(os.path.join(val_dir, img_dir_name)):
            img_path = os.path.join(val_dir, img_dir_name, img_name)
            print('这张是:', img_dir_name)
            search_name = search_img2vector(img_path)

            total_num += 1
            if img_dir_name in search_name:
                yes_num += 1
            else:
                print('❌')

            print('==' * 20)

    print('total_num:', total_num)
    print('yes_num:', yes_num)
    print(yes_num / total_num)
