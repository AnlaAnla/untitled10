import torch
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(3, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.fc4 = torch.nn.Linear(128, 64)
        self.fc5 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = F.selu(self.fc3(x))
        x = F.selu(self.fc4(x))
        x = self.fc5(x)
        return x


def get_data():
    # 生成一些随机数据作为训练集
    num_samples = 10
    length = torch.randint(1, 20, (num_samples,), dtype=torch.float)  # 长度范围0-10
    width = torch.randint(1, 20, (num_samples,), dtype=torch.float)  # 宽度范围0-5
    height = torch.randint(1, 20, (num_samples,), dtype=torch.float)  # 高度范围0-3
    volume = length * width * height  # 计算体积作为标签

    # 将输入数据和标签组合成PyTorch张量
    inputs = torch.stack([length, width, height], dim=1)
    labels = volume.unsqueeze(1)
    return inputs, labels


if __name__ == '__main__':
    model = torch.load("model.pt")

    inputs, labels = get_data()
    output = model(inputs)

    print("输入: ", inputs)
    print("标签: ", torch.reshape(labels, (labels.shape[0],)))
    print("预测: ", torch.reshape(output, (output.shape[0],)))
