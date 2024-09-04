import cv2
import os
from uuid import uuid4
from tqdm import tqdm
import numpy as np
import random


# 将mask贴到img_block上, 此处img_block和 mask 长,宽一致
def paste_mask(img_block, mask):
    img_block[np.where((mask != [255, 255, 255, 0]).all(axis=2))] = 0
    # 获取Alpha通道
    alpha_channel = mask[:, :, 3]
    # 将Alpha值为0的区域对应的RGB值设置为0
    mask[alpha_channel == 0] = [0, 0, 0, 0]
    mask = mask[:, :, :3]

    result = img_block + mask
    return result

    # # 显示结果
    # cv2.imshow("img", img_copy)
    # cv2.imshow('Result', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# mask比img小, 将mask随机贴到图片某个区域, 知乎返回 yolo格式坐标
def random_area(img_path, mask_path):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    block_x1 = random.randint(0, img.shape[1] - mask.shape[1])
    block_x2 = mask.shape[1] + block_x1

    block_y1 = random.randint(0, img.shape[0] - mask.shape[0])
    block_y2 = mask.shape[0] + block_y1
    img_block = img[block_y1:block_y2, block_x1:block_x2]

    result = paste_mask(img_block, mask)

    # 将图片原区域替换为 贴上mask的block
    img[block_y1:mask.shape[0] + block_y1, block_x1:mask.shape[1] + block_x1] = result

    # 生成 yolo 格式的坐标
    x = (block_x1 + block_x2) / (2 * img.shape[1])
    y = (block_y1 + block_y2) / (2 * img.shape[0])
    width = abs(block_x1 - block_x2) / img.shape[1]
    height = abs(block_y1 - block_y2) / img.shape[0]

    return img, x, y, width, height


def generate_scratch_yolo_data(img_dir_path, mask_dir_path,
                               train_save_dir_path, labels_save_dir_path):
    for img_name in tqdm(os.listdir(img_dir_path)):
        img_path = os.path.join(img_dir_path, img_name)

        for mask_name in os.listdir(mask_dir_path):
            mask_path = os.path.join(mask_dir_path, mask_name)

            # 贴图 保存图片
            img_mask, x, y, width, height = random_area(img_path, mask_path)

            train_img_name = uuid4().hex + '.jpg'
            img_save_path = os.path.join(train_save_dir_path, train_img_name)
            cv2.imwrite(img_save_path, img_mask)

            # 保存labels
            cls_id = 0
            yolo_label = f"{cls_id} {x} {y} {width} {height}\n"
            yolo_label_name = os.path.splitext(train_img_name)[0] + '.txt'
            yolo_label_path = os.path.join(labels_save_dir_path, yolo_label_name)

            with open(yolo_label_path, 'w', encoding='utf-8') as f:
                f.write(yolo_label)


if __name__ == '__main__':
    generate_scratch_yolo_data(
        img_dir_path=r"C:\Code\ML\Image\yolo_data02\Card_scratch\pokemon_not_scratch",
        mask_dir_path="C:\Code\ML\Image\yolo_data02\Card_scratch\scratch_mask",
        train_save_dir_path=r"C:\Code\ML\Image\yolo_data02\Card_scratch\generate_data01\images",
        labels_save_dir_path=r"C:\Code\ML\Image\yolo_data02\Card_scratch\generate_data01\labels"
    )
