import os

img_dir = r"C:\Code\ML\Image\yolov8_data\card_person_hand2\valid\images"
labels_dir = r"C:\Code\ML\Image\yolov8_data\card_person_hand2\valid\labels"

img_names = os.listdir(img_dir)
label_names = os.listdir(labels_dir)

for img_name in img_names:
    # print(os.path.splitext(img_name))
    if os.path.splitext(img_name)[0] + '.txt' in label_names:
        continue
    else:
        print(img_name)