from ultralytics import YOLO

model = YOLO(r"D:\Code\ML\Model\Card_Seg\yolov11n_card_seg01.pt")

result = model.predict(r"C:\Users\martin\Pictures\Alen Smailagic 2023-24 Panini Prizm EuroLeague #117 Orange #3849.jpg")
print(result)


print()

