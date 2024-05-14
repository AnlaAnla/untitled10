from ultralytics import YOLO

model = YOLO(r"C:\Code\ML\Model\Card_cls\yolo_handcard02_imgsz128.pt")

model.export(format='onnx', int8=True, half=True, simplify=True)