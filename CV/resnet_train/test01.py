import torch
from torchvision import models
# get list of models

import torch
# get list of models
# torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
#
# # load pretrained models, using ResNeSt-50 as an example
# net = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
net = models.resnet50(pretrained=False)
net.eval()
print(net)