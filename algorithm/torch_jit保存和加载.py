import torch.nn as nn
import torch.jit


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 28 * 28, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(-1, 16 * 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 创建模型实例
model = CustomModel()

# 将模型转换为 TorchScript 模型
scripted_model = torch.jit.script(model)

# 保存 TorchScript 模型
torch.jit.save(scripted_model, 'model.pt')

# 加载 TorchScript 模型
loaded_model = torch.jit.load('model.pt')
