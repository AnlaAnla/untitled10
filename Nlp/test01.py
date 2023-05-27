# -*- coding: utf-8 -*-
import os
import urllib
from bs4 import BeautifulSoup
import time
import requests


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
            img_thumbsSrc = soup.select('#srp-river-results > ul > li:nth-child({}) > div > div.s-item__image-section > '
                                        'div > a > div > img'.format(i))[0]['src']
            img_src = "https://i.ebayimg.com/images/g/" + img_thumbsSrc.split('/')[-2] + "/s-l500.jpg"
            img_urlList.append(img_src)
            print(i, img_src)

        except:
            print(i, "fault")

    return img_urlList


url = "https://www.ebay.com/sch/i.html?_from=R40&_trksid=p2334524.m570.l1313&_nkw=2013-14+panini+prizm&_sacat=0" \
      "&LH_TitleDesc=0&_odkw=2013-14+panini+prizm&_osacat=0&_ipg=240"


img_urlList = get_imgUrls(url=url)
print(len(img_urlList))

for i in range(len(img_urlList)):
    img_name = str(i) + ".jpg"
    save_path = os.path.join("D:\Code\ML\images\Mywork2\get_priz_13-14", img_name)

    try:
        img_file = requests.get(img_urlList[i]).content
        with open(save_path, 'wb') as f:
            f.write(img_file)
            print('save: ', i, ' ', img_name)
    except:
        print(i, "：fault")