from ultralytics import YOLO

model = YOLO(r"C:\Code\ML\Model\Card_Seg\yolov11n_card_seg01.pt")

def show(img_path):
    result = model.predict(img_path)
    result[0].show()



result = model.predict(r"C:\Code\ML\Image\ToMilvus\2024 card\2024, Court Kings, Acetate Rookies, 4, Alexandre Sarr, Washington Wizards\19ea2374-cfaf-4b0b-bd50-4f866f9713f6.jpg")
print(result)

print()
