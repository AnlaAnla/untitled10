import os

import cv2

def flip_img(dir_path):

    for filename in os.listdir(dir_path):
        if 'r' in filename:
            img_path = os.path.join(dir_path, filename)
            img = cv2.imread(img_path)
            img = cv2.flip(img, 1)
            cv2.imwrite(img_path, img)
    print('flip end')

def left2right(dir_path):
    for filename in os.listdir(dir_path):
        label_path = os.path.join(dir_path, filename)
        with open(label_path, 'r') as f:
            data = f.readline()
        angle = 180 - int(data)
        save_name = filename.replace('left', 'right')

        save_path = os.path.join(dir_path, save_name)
        with open(save_path, 'w') as f:
            f.write(str(angle))
            print(save_name, angle)
    print('left2right end')

def write_angle(dir_path, save_dir):
    for filename in os.listdir(dir_path):
        save_path = os.path.join(save_dir, filename.replace('jpg', 'txt'))
        with open(save_path, 'w', encoding='utf-8') as f:
            angle = input(filename + "Enter angle: ")
            f.write(angle)



dir_path = r'C:\Code\ML\Image\angle_data\test\label'
# flip_img(dir_path)
left2right(dir_path)

# write_angle(r"C:\Code\ML\Image\angle_data\test\img",
#             r"C:\Code\ML\Image\angle_data\test\label")