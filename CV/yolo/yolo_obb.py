from ultralytics import YOLO
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# model = YOLO(r"C:\Code\ML\Model\Card_cls\yolo_handcard02.pt")
model = YOLO(r"C:\Code\ML\Model\Card_Box\yolov8obb_card_box02.pt")
results = model.predict(r"C:\Code\ML\Image\Card_test\match_test\1 (36).jpg")
results[0].show()
print(results)
