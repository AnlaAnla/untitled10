from MyOnnxYolo import MyOnnxYolo
import matplotlib.pyplot as plt
import numpy as np
import cv2

cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

onnxYolo = MyOnnxYolo(r"C:\Code\ML\Model\onnx\yolov8n.onnx")
img_path = r"C:\Code\ML\Image\test02\a9.jpg"

img = onnxYolo.get_max_box(img_path)

print()