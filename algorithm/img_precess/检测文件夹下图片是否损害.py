import os
import glob
import time
import PIL.Image as Image
from multiprocessing.pool import ThreadPool

img_paths = glob.glob(r"C:\Code\ML\Image\card_cls\train_data6\train\*\*")
num = 0
bad_num = 0


def check(img_path):
    global num
    global bad_num

    if num < 31667:
        num += 1
        return
    try:
        img = Image.open(img_path)
        img.load()
        print("{}, {} ,{}".format(num, img.size, img_path))

        if img.size[0] < 224 or img.size[1] < 224:
            img.close()
            os.remove(img_path)
            print('<224， 删除！', img_path)
            bad_num += 1

    except Exception as e:
        print(f'读取图片失败，删除: {img_path}, \t错误信息: {e}')
        os.remove(img_path)
        bad_num += 1

    num += 1


if __name__ == '__main__':

    t1 = time.time()
    # pool = ThreadPool(4)
    # pool.map(check, img_paths)
    # pool.close()
    # pool.join()

    for path in img_paths:
        check(path)

    t2 = time.time()
    print('===========', num, '  time：', (t2 - t1), ' bad_num: ', bad_num)
