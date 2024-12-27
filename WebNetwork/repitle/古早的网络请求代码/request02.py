import requests
import os

def send_img(img_path):
    files = {'imgFile': open(img_path, 'rb')}

    response = requests.post(url, files=files)

    if response.status_code == 200:
        print('文件上传成功')
        print(f'服务器响应: {response.text}')
        print(f'服务器响应: {response.json()}')

    else:
        print(f'文件上传失败, 错误代码: {response.status_code}')


if __name__ == '__main__':
    url = "http://100.64.31.116:8080/image"

    # img_path = r"C:\Code\ML\Image\Card_test\test\324_4_0.jpg"
    # send_img(img_path)

    data_dir = r"D:\Code\ML\Image\ToMilvus\Card_series"
    for dir_name in os.listdir(data_dir):
        dir_path = os.path.join(data_dir, dir_name)
        print("=="*20)
        print(dir_name)
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            send_img(img_path)
