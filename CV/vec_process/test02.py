import os
import time
import cv2
from MyOnnxYolo import MyOnnxYolo

if __name__ == '__main__':
    onnxYolo = MyOnnxYolo(r"C:\Code\ML\Model\onnx\yolov10_handcard02_imgsz128.onnx")

    t1 = time.time()
    img = cv2.imread(r"C:\Code\ML\Image\Card_test\test02\a3.jpg")
    img = cv2.resize(img, (img.shape[0]//4, img.shape[1]//4))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
    t2 = time.time()
    onnxYolo.set_result(img, imgsz=128)

    t3 = time.time()
    onnxYolo.set_result(img, imgsz=128)

    t4 = time.time()
    onnxYolo.set_result(img, imgsz=128)

    t5 = time.time()
    img = cv2.imread(r"C:\Code\ML\Image\Card_test\test02\a5.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    t6 = time.time()
    onnxYolo.set_result(img, imgsz=128)

    print(f'读取图片: {t2 - t1}s')
    print(f'yolo检测: {t3 - t2}s')
    print(f'yolo检测: {t4 - t3}s')
    print(f'yolo检测: {t5 - t4}s')

    print(f'yolo检测: {time.time() - t6}s')
