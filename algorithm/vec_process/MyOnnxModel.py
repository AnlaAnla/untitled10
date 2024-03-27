import torch
import torchvision.models as models
import torchvision.transforms as transforms
import onnxruntime as rt
from PIL import Image
import numpy as np


class MyOnnxModel:
    def __init__(self, model_path: str) -> None:

        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]

        self.device = torch.device("cpu")

        self.model = rt.InferenceSession(model_path)

    def inference_transform(self):
        inference_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(self.norm_mean, self.norm_std),
        ])
        return inference_transform

    # 将向量降维
    def flattened_feature(self, input_features):
        # 进行全局平均池化操作
        pooled_features = np.mean(input_features, axis=(2, 3), keepdims=True)

        # 将输出展平为一维向量
        flattened_features = pooled_features.reshape(pooled_features.shape[0], -1)
        return flattened_features

    # 输入图片, 获取图片特征向量
    def run(self, img):
        if type(img) == type('path'):
            img = Image.open(img).convert('RGB')
        else:
            img = Image.fromarray(img)
            img = img.convert('RGB')
        transform = self.inference_transform()

        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze_(0).detach().numpy()

        input_features = self.model.run(None, {"input": img_tensor})[0]
        return self.flattened_feature(input_features)
