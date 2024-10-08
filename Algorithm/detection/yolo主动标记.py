import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("C:\Code\ML\Model\Card_cls\yolo_handcard01.pt", verbose=False)

train_dir = r'C:\Code\ML\Image\yolov8_data\mydataset.v5i.yolov8-obb\train\images'
label_dir = r'C:\Code\ML\Image\yolov8_data\mydataset.v5i.yolov8-obb\train\labels3'


# result = model.predict(r"C:\Code\ML\Image\yolov8_data\card_person_hand\images\2024_03_05 10_44_12.mp4-10500.jpg")
# print(result[0].show())
# print(result[0].boxes.cls == 0)
# print(result[0].boxes.xywhn)
# print(result[0])

def add_label(img_path, label_path):
    results = model.predict(img_path, verbose=False)
    length = len(results[0].boxes)
    if length == 0:
        return
    boxes = results[0].boxes

    for i in range(length):
        # if boxes[i].cls == 0:
            # print('1 {} {} {} {}\n'.
            #             format(boxes[i].xywhn[0][0], boxes[i].xywhn[0][1], boxes[i].xywhn[0][2], boxes[i].xywhn[0][3]))

        with open(label_path, 'a', encoding='utf-8') as f:
            print(i, label_path)
            f.write('{} {} {} {} {}\n'.
                    format(int(boxes[i].cls), boxes[i].xywhn[0][0], boxes[i].xywhn[0][1], boxes[i].xywhn[0][2], boxes[i].xywhn[0][3]))


img_num = 0
for img_name in os.listdir(train_dir):
    label_name = os.path.splitext(img_name)[0] + '.txt'

    img_path = os.path.join(train_dir, img_name)
    label_path = os.path.join(label_dir, label_name)

    img_num += 1
    print(img_num, img_name)

    add_label(img_path, label_path)
    print('_' * 30)

# 删除label为空的
# for label_name in os.listdir(label_dir):
#     label_path = os.path.join(label_dir, label_name)
#     # print(label_name, ': ', os.path.getsize(label_path))
#     if os.path.getsize(label_path) == 0:
#         img_name = os.path.splitext(label_name)[0]
#         img_path = os.path.join(train_dir, img_name)
#
#         if os.path.exists(img_path):
#             print(label_path, '----> ', img_path)
#             os.remove(img_path)
#             os.remove(label_path)
