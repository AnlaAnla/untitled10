import os
from util.MyOnnxModel_Resnet50 import MyOnnxModel
from util.MyOnnxYolo import MyOnnxYolo
# from util.MyBatchModel_Vit import MyViTFeatureExtractor
# from MyOnnxModel_MobileNetV3 import MyOnnxModel
from PIL import Image
import numpy as np
import cv2


# def normalize_numpy(vectors):
#     """
#     使用 NumPy 进行 L2 归一化。
#     """
#     if vectors.ndim == 1:  # 如果是单个向量
#         return vectors / np.linalg.norm(vectors)
#     elif vectors.ndim == 2:  # 如果是多个向量
#         norms = np.linalg.norm(vectors, axis=1, keepdims=True)
#         return vectors / norms
#     else:
#         raise ValueError("输入向量的维度必须是 1 或 2。")


# def vec_distance(vec1, vec2):
#     return np.linalg.norm(vec1 - vec2)

def cosine_similarity(vec1, vec2):
    """计算两个已 L2 归一化向量的余弦相似度。"""
    vec2 = vec2.flatten()  # 将vec2展开成1维

    return np.dot(vec1, vec2)


def add_img2vector(img_path, img_name):
    global img_id, vec_data, name_list

    output = onnxModel.run([img_path], normalize=True)
    # output = normalize_numpy(output)  # 归一化

    img_id += 1
    name_list.append(img_name)
    vec_data = np.concatenate([vec_data, output], axis=0)
    print('添加向量: ', img_id, img_name)


def search_img2vector(img, top_k: int = None):
    '''

        :param img:
        :param top_k: 如果为None, 返回最大的, 如果有数字, 返回排行前 top_k的
        :return:
    '''
    global img_id, vec_data, name_list

    output = onnxModel.run([img], normalize=True)
    # output = normalize_numpy(output)  # 归一化

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


def main():
    # onnxModel = MyViTFeatureExtractor(r"D:\Code\ML\Model\Card_cls\vit-base-patch16-224-AllCard08")
    onnxModel = MyOnnxModel(r"D:\Code\ML\Model\onnx\resnest50_AllCard08.onnx")
    vec_data = np.zeros((1, 768))

    # onnxYolo_card = MyOnnxYolo(r"C:\Code\ML\Model\onnx\yolo_card03.onnx")
    # vec_data = np.zeros((1, 2048))

    data_dir = r"D:\Code\ML\Image\_TEST_DATA\Card_test\mosic_prizm\prizm_yolo\base19-20 database"
    val_dir = r"D:\Code\ML\Image\_TEST_DATA\Card_test\mosic_prizm\prizm_yolo\base 19-20 val"

    name_list = ['background']

    img_id = 0

    for img_dir_name in os.listdir(data_dir):
        for img_name in os.listdir(os.path.join(data_dir, img_dir_name)):
            img_path = os.path.join(data_dir, img_dir_name, img_name)
            add_img2vector(img_path, img_dir_name)
    print('向量库建立完成')

    # 进行准确率测试
    total_num = 0
    yes_num = 0

    for img_dir_name in os.listdir(val_dir):
        for img_name in os.listdir(os.path.join(val_dir, img_dir_name)):
            img_path = os.path.join(val_dir, img_dir_name, img_name)

            search_name = search_img2vector(img_path)

            total_num += 1

            if search_name == img_dir_name:
                yes_num += 1
            else:
                print('❌', img_path)
            print(total_num, '\t这张是:', img_dir_name, '\t查询结果: ', search_name)
            print('==' * 20)
            print()

    print('total_num:', total_num)
    print('yes_num:', yes_num)
    print(yes_num / total_num)

def img_list_test():
    global onnxModel
    onnxModel = MyOnnxModel(r"C:\Code\ML\Model\onnx\resent_out17355_AllCard08.onnx")
    vec_data = np.zeros((1, 2048))
    name_list = []

    onnxYolo_card = MyOnnxYolo(r"C:\Code\ML\Model\Card_Seg\yolov11n_card_seg01.pt")

    data_dir = r"C:\Code\ML\Image\_TEST_DATA\Card_test02\vec_error\data"
    test_dit = r"C:\Code\ML\Image\_TEST_DATA\Card_test02\vec_error\test"

    # onnxYolo_card.set_result(r"C:\Code\ML\Image\_TEST_DATA\Card_test02\vec_error2\data\3.jpg")
    # img_bgr = onnxYolo_card.get_max_img(0)
    # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


    for img_name in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_name)

        onnxYolo_card.set_result(img_path)
        img_bgr = onnxYolo_card.get_max_img(0)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        cv2.imwrite(f"temp/data_{img_name}", img_bgr)
        output = onnxModel.run(img_rgb)

        name_list.append(img_name)
        vec_data = np.concatenate([vec_data, output], axis=0)
        print('添加向量: ', img_name)

    for img_name in os.listdir(test_dit):
        img_path = os.path.join(test_dit, img_name)

        onnxYolo_card.set_result(img_path)
        img_bgr = onnxYolo_card.get_max_img(0)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        cv2.imwrite(f"temp/test_{img_name}", img_bgr)
        output = onnxModel.run(img_rgb)

        distances = np.apply_along_axis(cosine_similarity, 1, vec_data, output)
        print(f"{img_name}: {distances}")



if __name__ == '__main__':
    # main()
    img_list_test()

'''
D:\Code\ML\Image\_TEST_DATA\Card_test\mosic_prizm\prizm_yolo\base19-20 database


'''
