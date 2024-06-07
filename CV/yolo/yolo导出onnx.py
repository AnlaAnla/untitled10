from ultralytics import YOLO

model = YOLO(r"C:\Code\ML\Model\Card_cls\yolov10_card_4mark_01.pt")

# model.export(format='onnx', int8=True, half=True, simplify=True)
model.export(format='onnx')