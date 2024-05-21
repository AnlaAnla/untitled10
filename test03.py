from ultralytics import YOLO

modle = YOLO(r"C:\Code\ML\Model\Card_cls\yolo_handcard02.pt")
img_path = r"C:\Code\ML\Image\Card_test\test\23.jpg"

results = modle.predict(img_path)


print(results)