import torch
import torch.nn as nn
import torchvision.models as models

model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 123)

for param in model.parameters():
    print(param.requires_grad)


dict_name = list(model.state_dict())
for i, p in enumerate(dict_name):
    print(i, p)