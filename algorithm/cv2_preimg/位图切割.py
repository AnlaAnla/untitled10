import PIL.Image
import scipy.misc
import numpy as np

# 获取第 n 个位平面
# 1-7
flat = 6


# 位图切割
def convert_2d(r):
    s = np.empty(r.shape, dtype=np.uint8)
    for j in range(r.shape[0]):
        for i in range(r.shape[1]):
            bits = bin(r[j][i])[2:].rjust(8, '0')
            fill = int(bits[-flat - 1])
            s[j][i] = 255 if fill else 0
    return s


im = PIL.Image.open(r"C:\Code\ML\Image\Card_test\test03\458a7303-e33a-4e3b-8626-156ea3269e6b_FRONT_MAIN.jpg")
im = im.convert('L')
im_mat = np.asarray(im)
im_conveted_mat = convert_2d(im_mat)

im_conveted = PIL.Image.fromarray(im_conveted_mat)
im_conveted.show()