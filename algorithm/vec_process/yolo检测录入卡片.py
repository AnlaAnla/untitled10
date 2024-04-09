import os
import time

import cv2

from MyOnnxModel_MobileNetV3 import MyOnnxModel
from MyOnnxYolo import MyOnnxYolo
import numpy as np


# 欧式距离
def vec_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


# 判断图像清晰度
def judge_clear(card_img):
    gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    # 计算拉普拉斯算子的方差, 判断图像的模糊程度
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f'卡片图像 拉普拉斯算子的方差: {fm}', end=' \t')
    if fm > 500:
        print(" --------------- 图片清晰")
        return True
    else:
        print("XXXXXX 图片模糊")
        return False


# 将图片转化为向量, 并与已经储存的向量进行对比
def add_img2vector(img):
    global img_id, vec_data, name_list, temp_array, frame

    onnxYolo_card.set_result(img)
    input_array = onnxYolo_card.get_max_img(cls_id=0)
    frame = onnxYolo_card.results[0].plot()
    # 拒绝不清晰的特征图像
    if not judge_clear(input_array):
        return False

    output = onnxModel.run(input_array)



    distances = np.apply_along_axis(vec_distance, 1, vec_data, output)
    # 向量距离对比,判断是否重复
    min_dis = np.min(distances)

    print('_' * 20)
    print(min_dis)
    if min_dis > 11:
        img_id += 1
        name_list.append(img_id)
        vec_data = np.concatenate([vec_data, output], axis=0)

        temp_array = input_array
        print('yes: ', img_id)
        return False
    print("No!: ", img_id, '\t重复id:', name_list[np.argmin(distances)])
    return True


if __name__ == '__main__':

    onnxModel = MyOnnxModel(r"C:\Code\ML\Model\onnx\model_features_card06.onnx")
    onnxYolo_card = MyOnnxYolo(r"C:\Code\ML\Model\onnx\yolo_card03.onnx")

    temp_array = np.random.rand(300, 300, 3)
    # vec_data = onnxModel.run(background_img_path)
    vec_data = np.zeros((1, 960))
    name_list = [0]

    img_id = 0

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        time.sleep(0.2)

        if ret:
            # cv2.imshow('frame', frame)
            if add_img2vector(frame):
                cv2.putText(frame, str(img_id), (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "NO!!", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)

            cv2.imshow("frame", frame)
            cv2.imshow('detect', temp_array)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print('not ret')
            break

    cap.release()
    cv2.destroyAllWindows()