from ultralytics import YOLO

model = YOLO(r"D:\Code\ML\Model\Card_Seg\yolov11n_card_seg01.pt")

# model.export(format='onnx', int8=True, half=True, simplify=True)
model.export(format='onnx')

