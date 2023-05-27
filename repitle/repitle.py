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


def get_imgUrls(url, num=189):
    img_urlList = []

    info = get_content(url)
    # print (info)
    soup = BeautifulSoup(info, 'html.parser')  # 设置解析器为“lxml”

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


def download_yearImg_thread(year, num, num_processes=4):

    year_dir = os.path.join(save_dir, year)
    # 年份文件夹是否存在，不存在则创建
    if not os.path.exists(year_dir):
        os.makedirs(year_dir)

    # 获取图片url列表
    url = "https://www.ebay.com/sch/i.html?_from=R40&_trksid=p2334524.m570.l1313&_nkw=20{}+Optic&_sacat=0&LH_TitleDesc=0&_odkw=2019-20+Optic&_osacat=0&_ipg=240".format(year)
    print("download ", url)

    img_urlList = get_imgUrls(url=url, num=num)
    print(len(img_urlList))

    # 线程池
    pool = ThreadPool(processes=num_processes)

    for i in range(len(img_urlList)):
        img_name = str(i) + ".jpg"
        save_path = os.path.join(year_dir, img_name)
        img_url = img_urlList[i]

        pool.apply_async(func=download_img, args=(img_url, save_path))
        # download_img(img_url, save_path)
        print('save: ', i, ' ', img_name)


if __name__ == '__main__':
    for year in down_years:
        download_yearImg_thread(year, num=200, num_processes=8)
        print('end')
