import torch
import onnx
import onnxruntime
import numpy as np
import time
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import PIL.Image as Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
data_phase = ['train', 'val']

if __name__ == '__main__':
    img_path = r"C:\Users\Administrator\Pictures\意味深长\v2-495dd342c78dbf0d056d8885c64052e4_r.jpeg"
    img = Image.open(img_path).convert("RGB")
    # img_train = data_transforms['train'](img)
    img_val = data_transforms['val'](img)
    img_val = img_val.unsqueeze(0)

    # 创建一个预测器会话喵~
    session = onnxruntime.InferenceSession("resnet50.onnx")

    t1 = time.time()

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: img_val.numpy()})

    t2 = time.time()

    model = models.resnet50()
    t3 = time.time()
    res = model(img_val)
    t4 = time.time()
    print('time01: ', t2 - t1)
    print(t4 - t3)
