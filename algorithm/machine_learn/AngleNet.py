import torch
import torch.nn as nn

class AngleNet(nn.Module):
    def __init__(self):
        super(AngleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16,kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,32,kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32*8*8, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32*8*8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x