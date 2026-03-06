from ultralytics import YOLO

model = YOLO(r"C:\Code\ML\Model\Card_Seg\yolov11n_card_seg01.pt")




result = model.predict(
    r"C:\Code\ML\Project\CardVideoSummary\static\frames\1ebb3bbe-8f97-4d5b-8740-bb2078227b46_12211009.jpg")
# print(result)

result[0].show()
print()

