import os.path
import pandas as pd
import cv2
import requests


def send_img(img_path):
    files = {'imgFile': open(img_path, 'rb')}

    response = requests.post(url, files=files)

    if response.status_code == 200:
        return response.json()
        # print('文件上传成功')
        # print(f'服务器响应: {response.text}')
        # print(f'服务器响应: {response.json()}')
    else:
        print(f'{img_path} 文件上传失败, 错误代码: {response.status_code}')
        return None


def download_img(save_path, url):
    try:
        response = requests.get(url)

        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {img_name}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")


if __name__ == '__main__':
    url = "http://100.64.1.9:8080/image"
    nas_url = 'http://100.64.0.31:9000/'

    data_dir = r"C:\Code\ML\Image\Card_test\psa_data2"
    csv_path = r"C:\Code\ML\Text\psa2_distill.csv"

    csv_data = pd.read_csv(csv_path)

    total_num = 0
    yes_num = 0
    no_similar_num = 0
    for img_name in os.listdir(data_dir)[:]:
        img_path = os.path.join(data_dir, img_name)

        response_json = send_img(img_path)
        total_num += 1

        img_index = int(os.path.splitext(img_name)[0]) - 1
        real_tag = csv_data[csv_data.columns[0]][img_index]
        if real_tag in response_json['tag'] or response_json['tag'] in real_tag:
            yes_num += 1
        else:
            print(f"\n-----------错误图片: {img_name}:{real_tag} ,  -----  识别结果: {response_json['tag']}  ,"
                  f"\npath:{response_json['path']},  \tscore:{response_json['score']}\n")

            if response_json['score'] > 0.3:
                print('---------不相似---------\n\n')
                no_similar_num += 1

            # 保存相似但名称错误的图片
            # else:
            #     img1 = cv2.imread(img_path)
            #     cv2.imwrite(os.path.join(r"C:\Code\ML\Image\Card_test\psa_source", img_name), img1)
            #
            #     img2_url = nas_url + response_json['path']
            #     img2_save_path = os.path.join(r"C:\Code\ML\Image\Card_test\psa_source_compare", img_name)
            #     download_img(img2_save_path, img2_url)
            #     print('----相似-----\n')

        print(f"上传: {img_name}, --- {yes_num}/{total_num}")
    print(f'共: {total_num}, 正确数: {yes_num},  识别错误且不相似的: {no_similar_num}')
    print('准确率: ', yes_num / total_num)
