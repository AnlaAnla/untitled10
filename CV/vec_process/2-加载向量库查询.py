import os
from util.MyOnnxModel_Resnet50 import MyOnnxModel
from Tool.MyOnnxYolo import MyOnnxYolo
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


# def vec_distance(vec1, vec2):
#     return np.linalg.norm(vec1 - vec2)

def cosine_similarity(vec1, vec2):
    """计算两个已 L2 归一化向量的余弦相似度。"""
    vec2 = vec2.flatten()  # 将vec2展开成1维

    return np.dot(vec1, vec2)


# def add_img2vector(img_path, img_name):
#     global img_id, vec_data, name_list
#
#     output = onnxModel.run(img_path)
#     output = normalize_numpy(output)  # 归一化
#
#     img_id += 1
#     name_list.append(img_name)
#     vec_data = np.concatenate([vec_data, output], axis=0)
#     print('添加向量: ', img_id, img_name)

def search_img2vector(img, top_k:int=None):
    '''

    :param img:
    :param top_k: 如果为None, 返回最大的, 如果有数字, 返回排行前 top_k的
    :return:
    '''
    global img_id, vec_data, name_list

    output = onnxModel.run(img)
    output = normalize_numpy(output)  # 归一化

    distances = np.apply_along_axis(cosine_similarity, 1, vec_data, output)

    if top_k is None:
        max_dis = np.max(distances)
        search_name = name_list[np.argmax(distances)]

        print(search_name, ': ', max_dis)
        return search_name
    else:
        search_names_dis = []
        if len(distances) < top_k:
            top_k = len(distances)

        # 获取最相似的k个元素
        partitioned_indices = np.argpartition(distances, -top_k)
        top_k_indices_arr = partitioned_indices[-top_k:][np.argsort(distances[partitioned_indices[-top_k:]])][::-1]
        for i in top_k_indices_arr:
            search_names_dis.append([str(name_list[i]), float(distances[i])])
        return search_names_dis


if __name__ == '__main__':
    onnxModel = MyOnnxModel(r"D:\Code\ML\Model\Card_cls2\resnest50_series05.onnx")
    onnxYolo_card = MyOnnxYolo(r"D:\Code\ML\Model\Card_Seg\yolov11n_card_seg01.engine")

    vec_data = np.load("temp/vec_data_series01.npy")
    name_list = np.load("temp/vec_data_series02_names.npy")

    print('加载向量库, 和名称库')

    val_dir = r"D:\Code\ML\Image\_TEST_DATA\Card_series_cls\2023-24"

    # for img_dir_name in os.listdir(val_dir):
    #
    #     sub_img_num = 0
    #     for img_name in os.listdir(os.path.join(val_dir, img_dir_name)):
    #         img_path = os.path.join(val_dir, img_dir_name, img_name)
    #
    #         onnxYolo_card.set_result(img_path)
    #         img = onnxYolo_card.get_max_img(cls_id=0)
    #         search_name = search_img2vector(img)
    #
    #         sub_img_num += 1
    #         new_path = os.path.join(val_dir, img_dir_name, f"{sub_img_num}-{search_name}.jpg")
    #         os.rename(img_path, new_path)
    #
    #         print('new_path: ', new_path)
    #         print('--'*30)

    # yes_num = 0
    onnxYolo_card.set_result(r"D:\Code\ML\Image\_TEST_DATA\Card_series_cls\2023-24\2023-24 CROWN ROYALE\4-2023-24 PANINI CROWN ROYALE BASE.jpg")
    img = onnxYolo_card.get_max_img(cls_id=0)
    search_names_dis = search_img2vector(img, top_k=10)
    print(search_names_dis)

