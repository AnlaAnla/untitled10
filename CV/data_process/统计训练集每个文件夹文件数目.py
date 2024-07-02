import glob
import os
import cv2
from CV.vec_process.MyOnnxYolo import MyOnnxYolo



father_dir = r"C:\Code\ML\Image\card_cls2\Series_cls01\train"

for sub_dir_name in os.listdir(father_dir):
    sub_dir_path = os.path.join(father_dir, sub_dir_name)

    print(f"{sub_dir_name:<30}: {len(os.listdir(sub_dir_path))}")
    # for img_name in os.listdir(sub_dir_path):
    #     img_path = os.path.join(sub_dir_path, img_name)
    #     if not img_name.endswith(".jpg"):
    #         print(img_path)
    #         os.remove(img_path)