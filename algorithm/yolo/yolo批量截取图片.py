import os
import PIL.Image as Image
import matplotlib.pyplot as plt
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from algorithm.vec_process.MyOnnxYolo import MyOnnxYolo


if __name__ == '__main__':
    model = MyOnnxYolo(r"C:\Code\ML\Model\yolo_card03.pt")

    train_dir = r"C:\Code\ML\Image\card_cls\train_data6_224\train"

    img_num = 0
    for img_dir in os.listdir(train_dir):
        for img_name in os.listdir(os.path.join(train_dir, img_dir)):
            img_path = os.path.join(train_dir, img_dir,img_name)

            # 排除224*224
            img = Image.open(img_path).convert('RGB')
            img_num += 1
            if img.size[0] == img.size[1] == 224:
                continue

            model.set_result(img)
            yolo_img = model.get_max_img(cls_id=0)
            img_yolo_224 = cv2.resize(yolo_img, (224, 224))
            cv2.imwrite(img_path, img_yolo_224)

        print(img_num, img_dir, ': yes')
    print('end')