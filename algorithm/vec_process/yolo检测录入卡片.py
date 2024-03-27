import os
import time

import cv2

from MyOnnxModel import MyOnnxModel
from MyOnnxYolo import MyOnnxYolo
import numpy as np


# 欧式距离
def vec_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

# 将图片转化为向量, 并与已经储存的向量进行对比
def add_img2vector(img):
    input_array = onnxYolo.get_max_box(img)
    output = onnxModel.run(input_array)

    global img_id, vec_data, name_list, temp_array


    distances = np.apply_along_axis(vec_distance, 1, vec_data, output)
    # 向量距离对比,判断是否重复
    min_dis = np.min(distances)

    print('_'*20)
    print(min_dis)
    if min_dis > 15:
        img_id += 1
        name_list.append(img_id)
        vec_data = np.concatenate([vec_data, output], axis=0)

        temp_array = input_array
        print('yes: ', img_id)
        return False
    print("No!: ", img_id)
    return True


if __name__ == '__main__':

    background_img_path = r"C:\Users\wow38\Pictures\Love\116669971_p0_master1200.jpg"
    img_dir = r"C:\Code\ML\Image\test02"

    onnxModel = MyOnnxModel(r"C:\Code\ML\Model\onnx\model_features_card03.onnx")
    onnxYolo = MyOnnxYolo(r"C:\Code\ML\Model\onnx\yolov8n.onnx")

    temp_array = np.random.rand(300, 300, 3)
    vec_data = onnxModel.run(background_img_path)
    name_list = [0]

    img_id = 0

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        time.sleep(0.2)

        if ret:
            # cv2.imshow('frame', frame)
            if add_img2vector(frame):
                cv2.putText(frame, str(img_id), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "NO!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)

            cv2.imshow("frame", frame)
            cv2.imshow('detect', temp_array)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print('not ret')
            break

    cap.release()
    cv2.destroyAllWindows()
