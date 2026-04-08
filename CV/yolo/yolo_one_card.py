from ultralytics import YOLO

model = YOLO(r"C:\Code\ML\Model\Card_Seg\yolov11n_card_seg01.pt")




result = model.predict(
    r"C:\Code\ML\Project\PokemonCardSearch\temp_yolo.jpg")
# print(result)

result[0].show()
print()

