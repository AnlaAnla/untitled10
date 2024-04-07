import os
import PIL.Image as Image
import matplotlib.pyplot as plt
import cv2

from algorithm.vec_process.MyOnnxYolo import MyOnnxYolo


if __name__ == '__main__':
    # model = MyOnnxYolo(r"C:\Code\ML\Model\yolo_card03.pt")

    train_dir = r'C:\Code\ML\Image\card_cls\one_piece'

    for img_dir in os.listdir(train_dir):
        for img_name in os.listdir(os.path.join(train_dir, img_dir)):
            img_path = os.path.join(train_dir, img_dir, img_name)

            # 排除224*224
            # img = Image.open(img_path)
            # if img.size[0] == img.size[1] == 224:
            #     continue

            # save_path = os.path.join(train_dir, img_dir, os.path.splitext(img_path)[0] + '_yolo.jpg')
            #
            # model.set_result(img_path)
            # max_img = model.get_max_img(cls_id=0)

            # img_224 = cv2.resize(result[0].orig_img, (224, 224))
            # img_yolo_224 = cv2.resize(max_img, (224, 224))
            img = cv2.imread(img_path)
            img_224 = cv2.resize(img, (224, 224))
            cv2.imwrite(img_path, img_224)
            # cv2.imwrite(save_path, img_yolo_224)

        print(img_dir, ': yes')
    print('end')