import os
import PIL.Image as Image
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from CV.vec_process.MyOnnxYolo import MyOnnxYolo


if __name__ == '__main__':
    model = MyOnnxYolo(r"C:\Code\ML\Model\onnx\yolo_handcard01.onnx")

    train_dir = r"C:\Code\ML\Image\Card_test\psa_source"

    img_num = 0
    for img_name in os.listdir(train_dir):
        img_path = os.path.join(train_dir, img_name)

        # 排除224*224
        img = Image.open(img_path).convert('RGB')
        img_num += 1
        if img.size[0] == img.size[1] == 224:
            continue

        model.set_result(img)
        yolo_img = model.get_max_img(cls_id=0)
        # img_yolo_224 = cv2.resize(yolo_img, (224, 224))
        cv2.imwrite(img_path, yolo_img)

        print(img_num, img_name, ': yes')
    print('end')