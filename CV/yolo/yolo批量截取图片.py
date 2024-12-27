import os
import PIL.Image as Image
import cv2
import glob
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from Tool.MyOnnxYolo import MyOnnxYolo

if __name__ == '__main__':
    model = MyOnnxYolo(r"D:\Code\ML\Model\onnx\yolo_handcard01.onnx")
    is_resize = False

    # train_dir = r"C:\Code\ML\Image\Card_test\psa_source"
    #
    # for img_name in os.listdir(train_dir):
    #     img_path = os.path.join(train_dir, img_name)
    img_paths = glob.glob(r"D:\Code\ML\Image\_CLASSIFY\card_cls2\2023 panini card set\train\*\*")

    img_num = 0
    for img_path in img_paths:
        # 排除224*224
        img = Image.open(img_path).convert('RGB')
        img_num += 1
        if img.size[0] == img.size[1] == 224 and is_resize:
            continue

        model.set_result(img)
        yolo_img = model.get_max_img(cls_id=0)
        if is_resize:
            yolo_img = cv2.resize(yolo_img, (224, 224))

        cv2.imwrite(img_path, yolo_img)

        print(img_num, os.path.split(img_path)[-1], ': yes')
    print('end')
