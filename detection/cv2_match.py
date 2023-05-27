import xml.etree.ElementTree as ET
import pickle
import os, cv2
import numpy as np
from os import listdir, getcwd
from os.path import join

postfix = 'jpg'
classes = []

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(xmlpath, xmlname):
    with open(xmlpath, "r", encoding='utf-8') as in_file:
        txtname = xmlname[:-4] + '.txt'
        txtfile = os.path.join('VOCdevkit/VOC2007/txt', txtname)
        tree = ET.parse(in_file)
        root = tree.getroot()
        filename = root.find('filename')
        img = cv2.imdecode(np.fromfile('VOCdevkit/VOC2007/JPEGImages/{}.{}'.format(xmlname[:-4], postfix), np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        res = []
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes:
                classes.append(cls)
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            res.append(str(cls_id) + " " + " ".join([str(a) for a in bb]))
        if len(res) != 0:
            with open(txtfile, 'w+') as f:
                f.write('\n'.join(res))


if __name__ == "__main__":
    xmlpath = 'VOCdevkit/VOC2007/Annotations'
    list = os.listdir(xmlpath)
    a = []
    for i in range(0, len(list)):
        try:
            path = os.path.join(xmlpath, list[i])
            if ('.xml' in path) or ('.XML' in path):
                convert_annotation(path, list[i])
                print('done', i)
            else:
                print('not xml file', i)
        except Exception as e:
            print(e)
            a.append(list[i])
    print(a)
    print(classes)