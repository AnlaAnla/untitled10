import torch
import cv2
import time
from torchvision import models
from utils.MyOnnxYolo import MyOnnxYolo
from torchvision import transforms
from PIL import Image
import os

model = torch.load(r"C:\Code\ML\Model\Card_cls2\resent_out29_Series01.pt")
model.eval()

yolo_model = MyOnnxYolo(r"C:\Code\ML\Model\onnx\yolo_handcard01.onnx")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

labels = ['2014-15 HOOPS', '2019-20 CONTENDERS', '2019-20 HOOPS PREMIUM', '2020-21 HOOPS', '2020-21 MOSAIC',
          '2020-21 PRIZM', '2021-22 DONRUSS', '2021-22 DONRUSS RATED ROOKIE', '2022-23 CONTENDERS', '2022-23 DONRUSS',
          '2022-23 DONRUSS OPTIC', '2022-23 PRIZM', '2023 ABSOLUTE FOOTBALL', '2023 BOWMAN BEST', '2023 CHINA SPORTS',
          '2023 GOOWIN', '2023 MARVEL', '2023 METAL', '2023 PRIZM FOOTBALL', '2023 TOPPS CHROME SOCCER',
          '2023-24 CONTENDERS', '2023-24 DONRUSS', '2023-24 DONRUSS ELITE', '2023-24 ORIGINS', '2023-24 PRIZM',
          '2023-24 SELECT CONCOURSE BLUE', '2023-24 SELECT CONCOURSE_', '2023-24 SELECT PREMIER',
          'POKEMEN']


def predict_cls(img):
    yolo_model.set_result(img)
    card_img = yolo_model.get_max_img(cls_id=0)
    card_img = Image.fromarray(card_img)

    img_tensor = transform(card_img)
    img_tensor = img_tensor.unsqueeze(0)

    # Perform prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs.data, 1)

    return labels[int(predicted)]


if __name__ == '__main__':
    # Load the image
    # image_path = r"C:\Code\ML\Image\Match_template\2024-07-03\2023 中国体育\2024_07_03___05_37_12.jpg"
    # img = Image.open(image_path).convert('RGB')

    # result = predict_cls(img)

    data_dir = r"C:\Code\ML\Image\Match_template\2024-07-03"
    yes_num = 0
    total = 0
    for img_dir_name in os.listdir(data_dir):
        img_dir_path = os.path.join(data_dir, img_dir_name)
        for img_name in os.listdir(img_dir_path):
            img_path = os.path.join(img_dir_path, img_name)
            img = Image.open(img_path).convert('RGB')
            cls_name = predict_cls(img)

            if img_dir_name == cls_name:
                yes_num += 1
                print(f'{img_dir_name}: 正确')
            else:
                print(f"{img_dir_name}: 错误")
            total += 1
            print(f"真实类: {img_dir_name}, 预测结果: {cls_name}")
        print("_"*20)
    print(f"{yes_num}/{total}")
