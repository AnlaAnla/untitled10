from ultralytics import YOLO
import os
import cv2

yolo = YOLO(r"C:\Code\ML\Model\Card_Box\yolov8_attention_card_scratch03.onnx")


def show(img_path):
    result = yolo.predict(img_path)
    result[0].show()


# show(r"C:\Code\ML\Image\Card_test\test03\8fac49640b61902b13538e8b228abfa.jpg")
# print()

save_dir_path = r"C:\Code\ML\Image\Card_test02\scratch_result"

no_scratch_path = r"C:\Code\ML\Image\Card_test02\scratch_no_test"
scratch_path = r"C:\Code\ML\Image\Card_test02\scratch_test"

for i, img_name in enumerate(os.listdir(no_scratch_path)):
    img_path = os.path.join(no_scratch_path, img_name)

    result = yolo.predict(img_path)
    img = result[0].plot()

    save_path = os.path.join(save_dir_path, f"no_{i}.jpg")
    cv2.imwrite(save_path, img)
    print(img_path)

for i, img_name in enumerate(os.listdir(scratch_path)):
    img_path = os.path.join(scratch_path, img_name)

    result = yolo.predict(img_path)
    img = result[0].plot()

    save_path = os.path.join(save_dir_path, f"have_{i}.jpg")
    cv2.imwrite(save_path, img)
    print(img_path)
