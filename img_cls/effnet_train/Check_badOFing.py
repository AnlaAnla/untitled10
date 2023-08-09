import os
import glob
import time
import PIL.Image as Image
from multiprocessing.pool import ThreadPool


num = 0
bad_num = 0

# 批量检查图片是否损坏, is_delete: 是否删除损坏图片，threshold_size: 是否删除长或宽小于该数值的图片
def check(img_path, is_delete=False, threshold_size=None):
    global num
    global bad_num
    try:
        img = Image.open(img_path)
        print("{}, {} ,{}".format(num, img.size, img_path))

        if threshold_size != None:
            if img.size[0] < 224 or img.size[1] < 224:
                img.close()
                os.remove(img_path)
                print('<， ', threshold_size, ',删除！ ', img_path)
                bad_num += 1

    except:
        print('读取图片失败，删除: ', img_path)
        # os.remove(img_path)
        bad_num += 1

    num += 1


if __name__ == '__main__':
    # 使用一下路径格式
    # r"D:\Code\ML\images\Mywork3\train_data4_224\train\*\*"

    source_dir = r"D:\Code\ML\images\Mywork3\train_data4_224\train\*\*"
    img_paths = glob.glob(source_dir)

    t1 = time.time()
    # 多线程用法
    # pool = ThreadPool(4)
    # pool.map_async(check, img_paths)

    # 遍历
    for path in img_paths:
        check(path)

    t2 = time.time()
    print('===========', num, '  time：', (t2 - t1), ' bad_num: ', bad_num)
