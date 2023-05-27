import xml.dom.minidom
import os

# 读取文件夹xml文件中类的名字和数量
label_path = r"D:\Code\ML\images\Mywork\work\VOCdevkit\dataset\labels"
name_list = []


for xml_name in os.listdir(label_path):
    #打开xml文档
    dom = xml.dom.minidom.parse(os.path.join(label_path, xml_name))
    root = dom.documentElement
    cc=dom.getElementsByTagName('object')

    for object in cc:
        name = object.getElementsByTagName('name')[0].firstChild.data

        if name == "PRIZM" or name == "mosaic":
            print(xml_name, "------------")

        name_list.append(name)


name_tuple = set(name_list)

for item in name_tuple:
    print("{} : {}".format(item, name_list.count(item)))

print("\nclasses: ", len(name_tuple))
print(name_tuple)