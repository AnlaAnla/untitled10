import os

import requests

def send_img(img_path):
    files = {'imgFile': open(img_path, 'rb')}

    response = requests.post(url, files=files)

    if response.status_code == 200:
        print(f'服务器响应: {response.json()}')
        return response.json()
    else:
        print(f'文件上传失败, 错误代码: {response.status_code}')

if __name__ == '__main__':
    url = "http://100.64.1.9:8080/image"

    # img_dir = r"C:\Code\ML\Image\Card_test\mosic_prizm\prizm_yolo\base 19-20 val\#7 Yao Ming"
    img_dir = r"C:\Code\ML\Image\Card_test\mosic_prizm\prizm\base 19-20 val\#6 Allen Iverson"
    yes_num = 0
    for i, img_name in enumerate(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_name)
        print(i)
        data = send_img(img_path)
        if "ALLEN IVERSON" not in data['tag']:
            print(f'{i} 错误')
            print(data)
        else:
            print(f'{i} yes!!!!!!!!')
            yes_num += 1
        print(f'{yes_num}/{i}')




