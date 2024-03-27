import os

from MyOnnxModel import MyOnnxModel
from MyOnnxYolo import MyOnnxYolo
import numpy as np


def vec_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def compare(n1, n2):
    print(name_list[n1], name_list[n2])
    print(vec_distance(vec_data[n1], vec_data[n2]))


if __name__ == '__main__':

    background_img_path = r"C:\Users\wow38\Pictures\Love\116669971_p0_master1200.jpg"
    img_dir = r"C:\Code\ML\Image\test02"

    onnxModel = MyOnnxModel(r"C:\Code\ML\Model\onnx\model_features_card03.onnx")
    onnxYolo = MyOnnxYolo(r"C:\Code\ML\Model\onnx\yolov8n.onnx")

    vec_data = onnxModel.run(background_img_path)
    name_list = ['background']

    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)

        input_array = onnxYolo.get_max_box(img_path)
        output = onnxModel.run(input_array)

        # add
        name_list.append(img_name)
        vec_data = np.concatenate([vec_data, output], axis=0)

    print(vec_data.shape)
    print(vec_distance(vec_data[0], vec_data[1]))

    new_vec = onnxModel.run(r"C:\Code\ML\Image\test02\a9.jpg")
    # 广播
    distances = np.apply_along_axis(vec_distance, 1, vec_data, new_vec)

    print()


