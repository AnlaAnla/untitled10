import torchvision
from thop import profile
import torch
import numpy as np
import torch.nn as nn


# 假设输入特征为 (batch_size, channels, height, width)
# input_features = torch.randn(1, 576, 7, 7)

def flattened_feature_numpy(input_features):
    # 进行全局平均池化操作
    pooled_features = np.mean(input_features, axis=(2, 3), keepdims=True)

    # 将输出展平为一维向量
    flattened_features = pooled_features.reshape(pooled_features.shape[0], -1)
    return flattened_features


def flattened_feature(input_features):
    # 定义全局平均池化层
    gap = nn.AdaptiveAvgPool2d(1)

    # 进行全局平均池化操作
    pooled_features = gap(input_features)

    # 将输出展平为一维向量
    flattened_features = pooled_features.view(pooled_features.size(0), -1)
    return flattened_features


# model = torchvision.models.mobilenet_v3_large(pretrained=False)
input1 = torch.randn(1, 3, 224, 224)
input2 = torch.randn(1, 3, 224, 224)

model = torchvision.models.mobilenet_v3_small(pretrained=False)
features = model.features

out1 = features(input1)
out2 = features(input2)

print(out1.shape)
print(out2.shape)
print()
