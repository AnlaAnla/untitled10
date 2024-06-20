from ultralytics import YOLO

model = YOLO(r"C:\Code\ML\Model\Card_Box\card_2box.pt")
results = model.predict(r"C:\Code\ML\Image\yolo_data02\Card_box\trian\5.jpg")
results[0].show()
print(results)