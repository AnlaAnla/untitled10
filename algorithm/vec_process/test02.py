from MyOnnxYolo import MyOnnxYolo
import matplotlib.pyplot as plt
import numpy as np
import cv2

onnxYolo = MyOnnxYolo(r"C:\Code\ML\Model\onnx\yolo_card03.onnx")
img_path = r"C:\Code\ML\Image\test02\22 (4).jpg"

img = onnxYolo.set_result(img_path)
results = onnxYolo.results
results[0].show()
