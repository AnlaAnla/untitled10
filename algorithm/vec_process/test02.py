import os

from MyOnnxYolo import MyOnnxYolo

if __name__ == '__main__':
    onnxYolo = MyOnnxYolo(r"C:\Code\ML\Model\onnx\yolo_handcard01.onnx")

    data_dir = r"C:\Code\ML\Image\Card_test\test03"

    for img_name in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_name)
        onnxYolo.set_result(img_path)
        results = onnxYolo.results

        results[0].show()

    print('end')