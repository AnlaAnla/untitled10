import os
import requests


# /home/martin/Pictures/ReverseImageSearch/OBJDetected/prizm 21-22/Widescreen/ #10 LEBRON JAMES/*.jpg

# 发送图片判读正确率
def send_img(img_path):
    files = {'imgFile': open(img_path, 'rb')}
    response = requests.post(url, files=files)

    if response.status_code == 200:
        data = response.json()
        # txt = data['tag'] + " -- " + data['detail']['player']
        return data

        # print('文件上传成功')
    else:
        return f'文件上传失败, 错误代码: {response.status_code}'


if __name__ == '__main__':
    url = "http://192.168.56.116:8088/image/"

    father_dir = r'C:\Code\ML\Image\Card_test\mosic_prizm\prizm\base 19-20 val'

    # data = send_img(
    #     r"C:\Code\ML\Image\Card_test\mosic_prizm\prizm\base 19-20 val\#254 Jaxson Hayes\fbff10f6dfc26a666798bd653aa1fc66.jpg")
    # print(data)

    total_num = 0
    yes_num = 0
    for dir_name in os.listdir(father_dir):
        for img_name in os.listdir(os.path.join(father_dir, dir_name)):
            img_path = os.path.join(father_dir, dir_name, img_name)

            # print(img_path)
            data = send_img(img_path)
            total_num += 1

            print(total_num)
            print(dir_name)
            print(data)

            real_id = dir_name.split(' ')[0].split('#')[-1]
            search_id = data['tag'].split('#')[-1]

            if real_id == search_id:
                yes_num += 1
            else:
                print('❌')

            # print(data)

            print("_" * 30)
            print('\n')

    print('总数: ', total_num)
    print('正确: ', yes_num)
    print('准确率: ', yes_num / total_num)
