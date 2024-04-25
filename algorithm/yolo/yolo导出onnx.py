from ultralytics import YOLO

model = YOLO(r"C:\Code\ML\Model\yolo_handcard01.pt")

model.export(format='onnx', int8=True, half=True, simplify=True)