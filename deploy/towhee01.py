import timm
import torch
from torchvision import transforms, models


# feacture = model.forward_features(torch.rand((1,3,224,224)))

model = models.efficientnet_b7()
print(model)