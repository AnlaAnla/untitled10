import requests
import os
from PIL import Image, ImageDraw


def send_img(url, img_path):
    files = {'imgFile': open(img_path, 'rb')}

    response = requests.post(url, files=files)

    if response.status_code == 200:
        return response.json()


if __name__ == '__main__':
    position_url = "http://192.168.31.116:8080/card_position"

    img_path = r"D:\Code\ML\Image\_TEST_DATA\Card_test\test03\f0ba2278b6b445ff823139564116b92.jpg"
    img = Image.open(img_path)

    data = send_img(position_url, img_path)
    draw = ImageDraw.Draw(img)
    draw.rectangle((data['x1'], data['y1'], data['x2'], data['y2']), outline='green', width=8)

    img.show()
