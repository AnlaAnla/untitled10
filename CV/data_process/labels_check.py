import json
import os


# def check_TxtLabels(labels_path):


def check_AnyLabels(labels_path):
    for label_name in os.listdir(labels_path):
        label_path = os.path.join(labels_path, label_name)
        with open(label_path, "r", encoding='utf-8') as f:
            data = json.load(f)

        for shape in data['shapes']:
            if len(shape['points']) > 2:
                print(f"异常 - {label_path}: {len(shape['points'])}")
        print('end')


anyLabels_path = r"C:\Code\ML\Image\yolo_data02\Card_scratch01\pre_data\0806\anylabels"

check_AnyLabels(anyLabels_path)