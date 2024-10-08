import torchvision
from thop import profile
import torch

model = torchvision.models.mobilenet_v3_large(pretrained=False)
input = torch.randn(1, 3, 224, 224)
macs, params = profile(model, inputs=(input, ))

print('macs = ' + str(macs/1000**3) + 'G')
print('Params = ' + str(params/1000**2) + 'M')