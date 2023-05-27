import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
图片信息查看网站
https://fotoforensics.com/

为防止手机拍照、截图、网站上传的图片被附加盲水印，所以用cv读取在保存，去信息化

下一步还需增加噪音，去除图片隐藏信息
'''
def gasuss_noise(image, mean=0.1, var=0.003):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out

def save_img(image_path, img):
    imageArray_path = image_path.split(".")
    savepath = imageArray_path[0] + '_privacy' + "." + imageArray_path[1]

    # save
    cv2.imwrite(savepath, img)
    print("保存完成")



if __name__=="__main__":
    image_path = r"D:\Pictures\Camera Roll\1a2d568e-0dbe-4fe0-a207-d3c46d383121.png"
    img = cv2.imread(image_path)

    # add noice
    img = gasuss_noise(img)
    save_img(image_path, img)

