import os
from CV.vec_process.util.MyOnnxModel_Resnet50 import MyOnnxModel
# from MyOnnxModel_MobileNetV3 import MyOnnxModel
import numpy as np


def normalize_numpy(vectors):
    """
    使用 NumPy 进行 L2 归一化。
    """
    if vectors.ndim == 1:  # 如果是单个向量
        return vectors / np.linalg.norm(vectors)
    elif vectors.ndim == 2:  # 如果是多个向量
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms
    else:
        raise ValueError("输入向量的维度必须是 1 或 2。")


def add_img2vector(img_path, img_name):
    global img_id, vec_data, name_list

    output = onnxModel.run(img_path)
    output = normalize_numpy(output)  # 归一化

    img_id += 1
    name_list.append(img_name)
    vec_data = np.concatenate([vec_data, output], axis=0)
    print('添加向量: ', img_id, img_name)



if __name__ == '__main__':
    onnxModel = MyOnnxModel(r"D:\Code\ML\Model\Card_cls2\resnest50_series05.onnx")
    # vec_data = np.zeros((1, 960))

    # onnxYolo_card = MyOnnxYolo(r"C:\Code\ML\Model\onnx\yolo_card03.onnx")
    vec_data = np.zeros((1, 2048))

    data_dir = r'D:\Code\ML\Image\_CLASSIFY\card_cls2\train_serices_cls_data_yolo224\val'
    # val_dir = r"D:\Code\ML\Image\_CLASSIFY\card_cls2\train_serices_cls_data_yolo224\train"

    name_list = ['background']

    img_id = 0

    for img_dir_name in os.listdir(data_dir):
        for img_name in os.listdir(os.path.join(data_dir, img_dir_name)):
            img_path = os.path.join(data_dir, img_dir_name, img_name)
            add_img2vector(img_path, img_dir_name)

    name_list = np.array(name_list)

    np.save("temp/vec_data_series01.npy", vec_data)
    np.save("temp/vec_data_series02_names.npy", name_list)
    print('向量库建立完成')

