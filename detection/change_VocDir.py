
import shutil
import os


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("creat!")
    else:
        print("have already existed")

if __name__ == '__main__':
    datasets_path = r"D:\Code\ML\images\Mywork\dataset"
    imgs_path = r"D:\Code\ML\images\Mywork\dataset\images"
    labels_path = r"D:\Code\ML\images\Mywork\dataset\labels"

    paths = [imgs_path, labels_path]
    names = ['train', 'test', 'valid']

    for name in names:
        mkdir(os.path.join(datasets_path, name))

    for name in names:
        shutil.move(os.path.join(imgs_path, name), os.path.join(datasets_path, name, "images"))

    for name in names:
        shutil.move(os.path.join(labels_path, name), os.path.join(datasets_path, name, "labels"))