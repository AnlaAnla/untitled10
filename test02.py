from ultralytics import YOLO

model = YOLO(r"C:\Code\ML\Model\onnx\yolov10_handcard02_imgsz128.onnx")

result = model.predict(r"C:\Code\ML\Image\Card_test\test02\5.jpg", imgsz=128)

result[0].show()