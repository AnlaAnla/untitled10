import os
import glob
import time
import PIL.Image as Image
from multiprocessing import Pool
from tqdm import tqdm


def check(img_path):
    global num
    global bad_num

    try:
        img = Image.open(img_path)
        img.load()
        img.close()
        # print("{}, {} ,{}".format(num, img.size, img_path))

        # if img.size[0] < 224 or img.size[1] < 224:
        #     img.close()
        #     os.remove(img_path)
        #     print('<224， 删除！', img_path)
        #     bad_num += 1

    except Exception as e:
        print(f'读取图片失败，删除: {img_path}, \t错误信息: {e}')
        img.close()
        # os.remove(img_path)
        bad_num += 1

    num += 1


if __name__ == '__main__':
    img_paths = glob.glob(r"D:\Code\ML\Image\_CLASSIFY\card_cls2\2023 panini card set\train\*\*")
    num = 0
    bad_num = 0

    t1 = time.time()

    for path in tqdm(img_paths):
        check(path)

    t2 = time.time()
    print('===========', num, '  time：', (t2 - t1), ' bad_num: ', bad_num)
