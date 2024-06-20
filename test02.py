import os

from ultralytics import YOLO
import numpy as np
import time
import cv2



model = YOLO(r"C:\Code\ML\Model\onnx\yolov10_card_4mark_01.onnx", task='detect')

data_path = r"C:\Code\ML\Image\Card_test\yolo_mark_test"

for img_name in os.listdir(data_path):
    img_path = os.path.join(data_path, img_name)
    results = model.predict(img_path)
    img = results[0].plot()

    cv2.imwrite(img_path, img)
    print(img_path)
print('end')