import json
import os
from tqdm import tqdm
'''
用Anylabeling标记的数据默认格式转化为yolo格式, 
classes_path为txt文件,每一行写类名称
'''


def AnyLabeling2Yolo(Anylabeling_path):

    with open(Anylabeling_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    imageHeight = data['imageHeight']
    imageWidth = data['imageWidth']

    _yolo_label = ''
    for shape in data['shapes']:
        if shape['shape_type'] == "polygon":
            cls_id = classes.index(shape['label'])
            point_list = []

            points = shape['points']
            for point in points:
                point_list.append(str(point[0] / imageWidth))
                point_list.append(str(point[1] / imageHeight))

            yolo_label_one = str(cls_id) + ' ' + ' '.join(point_list)
            _yolo_label = _yolo_label + yolo_label_one + '\n'
        elif shape['shape_type'] == "rectangle":
            cls_id = classes.index(shape['label'])

            points = shape['points']
            x = (points[0][0]+points[1][0])/(2*imageWidth)
            y = (points[0][1]+points[1][1])/(2*imageHeight)
            width = abs(points[0][0]-points[1][0])/imageWidth
            height = abs(points[0][1]-points[1][1])/imageHeight

            yolo_label_one = f"{cls_id} {x} {y} {width} {height}"
            _yolo_label = _yolo_label + yolo_label_one + '\n'
        else:
            print(f'未知格式!!! ----- Anylabeling_path: {shape["shape_type"]}')

    return _yolo_label


if __name__ == '__main__':
    classes_path = r"C:\Code\ML\Image\yolo_data02\Card_scratch\Scratch\classes.txt"
    classes = open(classes_path).read().splitlines()
    print(classes)

    AnyLabels_dir = r"C:\Code\ML\Image\yolo_data02\Card_scratch\Scratch\scratch\anylabesl"
    YoloLabels_save_dir = r"C:\Code\ML\Image\yolo_data02\Card_scratch\Scratch\scratch\labels"

    for AnyLabel_name in tqdm(os.listdir(AnyLabels_dir)):
        AnyLabel_path = os.path.join(AnyLabels_dir, AnyLabel_name)

        yolo_label = AnyLabeling2Yolo(AnyLabel_path)
        yolo_label_name = os.path.splitext(os.path.split(AnyLabel_path)[-1])[0] + '.txt'

        save_path = os.path.join(YoloLabels_save_dir, yolo_label_name)

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(yolo_label)

        # print(yolo_label_name)

    print("处理结束.")