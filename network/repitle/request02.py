import requests


def send_img(img_path):
    files = {'imgFile': open(img_path, 'rb')}

    response = requests.post(url, files=files)

    if response.status_code == 200:
        print('文件上传成功')
        print(f'服务器响应: {response.text}')
        print(f'服务器响应: {response.json()}')

        all_data = response.json()['tag']
        code, data = all_data.split(' ', 1)
        searchParam, team = data.split(' - ')
        score = response.json()['score']

        print({
            'all_data': all_data,
            'code': code,
            'team': team,
            'searchParam': searchParam,

            'vector_score': score
        })
    else:
        print(f'文件上传失败, 错误代码: {response.status_code}')


if __name__ == '__main__':
    url = "http://100.64.1.9:8080/image"

    img_path = r"C:\Code\ML\Image\Card_test\test\324_4_0.jpg"
    send_img(img_path)
