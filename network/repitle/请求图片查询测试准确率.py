import os.path

import requests


def send_img(img_path):
    files = {'imgFile': open(img_path, 'rb')}

    response = requests.post(url, files=files)

    if response.status_code == 200:
        return response.json()['tag']
        # print('文件上传成功')
        # print(f'服务器响应: {response.text}')
        # print(f'服务器响应: {response.json()}')
    else:
        print(f'{img_path} 文件上传失败, 错误代码: {response.status_code}')
        return None


if __name__ == '__main__':
    url = "http://100.64.1.9:8080/image"

    data_dir = r"C:\Code\ML\Image\Card_test\psa_data"

    total_num = 0
    yes_num = 0
    for img_name in os.listdir(data_dir)[:25]:
        img_path = os.path.join(data_dir, img_name)

        tag = send_img(img_path)
        total_num += 1
        if os.path.splitext(img_name)[0] == tag:
            yes_num += 1
        else:
            print(f"-----------错误图片: {img_name}, --- 识别结果: {tag}")

        print(f"上传: {img_name}, --- {yes_num}/{total_num}")
    print(f'共: {total_num}, 正确数: {yes_num}')
    print('准确率: ', yes_num/total_num)

