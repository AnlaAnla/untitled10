import torch


torch.hub.list('zhanghang1989/ResNeSt')
model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 100, bias=True)
model.load_state_dict(torch.load(r"D:\Code\ML\Model\Card_cls2\resnest50_PaniniCard03.pth"))

print(model)
