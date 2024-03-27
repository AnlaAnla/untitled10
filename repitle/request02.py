import requests

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
    url = "http://192.168.56.116:8088/image/"

    img_path = r"C:\Code\ML\Image\test02\1 (5).jpg"
    send_img(img_path)
