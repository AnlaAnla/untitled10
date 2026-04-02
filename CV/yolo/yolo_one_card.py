from ultralytics import YOLO

model = YOLO(r"C:\Code\ML\Model\Card_Seg\yolov11n_card_seg01.pt")




result = model.predict(
    r"C:\Code\ML\Image\_CLASSIFY\card_cls2\Pokemon01\宝可梦平行\平行闪\0d31e2cc3be4e933a20eac3f3dffb859.png")
# print(result)

result[0].show()
print()

