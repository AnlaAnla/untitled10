# -*- coding: utf-8 -*-
import os
import urllib
from bs4 import BeautifulSoup
from multiprocessing.pool import ThreadPool
import time
import requests

down_years = ['16-17', '17-18', '18-19', '19-20', '20-21', '21-22']
save_dir = r"D:\Code\ML\images\Mywork2\Url_O"


def get_content(url):
    '''
    @url:需要下载的网址
    下载网址
    '''
    html = urllib.request.urlopen(url)
    content = html.read().decode('utf-8')  # 转码
    html.close()  # 记得要将打开的网页关闭，否则会出现意想不到的问题
    return content


def get_imgUrls(url):
    img_urlList = []

    info = get_content(url)
    # print (info)
    soup = BeautifulSoup(info, 'html.parser')  # 设置解析器为“lxml”
    num = 0
    # 读取页面一的网址，为了获得大尺寸图片
    for i in range(1, num + 1):
        try:
            img_thumbsSrc = \
                soup.select('#srp-river-results > ul > li:nth-child({}) > div > div.s-item__image-section > '
                            'div > a > div > img'.format(i))[0]['src']
            img_src = "https://i.ebayimg.com/images/g/" + img_thumbsSrc.split('/')[-2] + "/s-l500.jpg"
            img_urlList.append(img_src)
            print(i, img_src)

        except:
            print(i, "fault")

    return img_urlList


def download_img(img_url, save_path):
    try:
        img_file = requests.get(img_url).content
        with open(save_path, 'wb') as f:
            f.write(img_file)
    except:
        print(save_path, ": fault")


def download_yearImg():

    # 获取图片url列表
    url = "https://onepiece-cardgame.dev/cards?f=%24R+%28col%3A%22%2F%22%29"
    print("download ", url)

    img_urlList = get_imgUrls(url=url)
    print(len(img_urlList))



    # for i in range(len(img_urlList)):
    #     img_name = str(i) + ".jpg"
    #     save_path = os.path.join(year_dir, img_name)
    #     img_url = img_urlList[i]
    #
    #     pool.apply_async(func=download_img, args=(img_url, save_path))
    #     # download_img(img_url, save_path)
    #     print('save: ', i, ' ', img_name)


if __name__ == '__main__':
    # download_yearImg()

    content = get_content("https://onepiece-cardgame.dev/cards?f=%24R+%28col%3A%22%2F%22%29")
    soup = BeautifulSoup(content, 'html.parser')
    print(content)