import requests
import os


def send_img(url, img_path):
    files = {'imgFile': open(img_path, 'rb')}

    response = requests.post(url, files=files)

    if response.status_code == 200:
        print('文件上传成功')
        print(f'服务器响应: {response.json()}')

    else:
        print(f'文件上传失败, 错误代码: {response.status_code}')


if __name__ == '__main__':
    info_url = "http://192.168.31.116:8080/image"
    position_url = "http://192.168.31.116:8080/card_position"

    img_path = r"D:\Code\ML\Image\_TEST_DATA\Card_test\test\5d368b89732b6a0e5312374ef05e4c61.jpg"

    send_img(info_url, img_path)
    send_img(position_url, img_path)
