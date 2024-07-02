import torch
from torchvision import models

model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(in_features=2048, out_features=51)

model.load_state_dict(torch.load(r"C:\Code\ML\Model\Card_cls2\resent_out51_Series01.pth",
                                 map_location='cpu'))
model.eval()

torch.save(model, r"C:\Code\ML\Model\Card_cls2\resent_out51_Series01.pt")