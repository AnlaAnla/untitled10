import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
from AngleNet import AngleNet


# 定义数据集
class AngleDataset(Dataset):
    def __init__(self, images, angles):
        self.images = images
        self.angles = angles
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        angle = self.angles[idx]
        image = self.transform(image)
        return image, float(angle)

def load_data(data_dir):
    images = []
    angles = []
    img_dir = os.path.join(data_dir, 'img')
    label_dir = os.path.join(data_dir, 'label')

    # 获取文件名列表
    img_files = sorted(os.listdir(img_dir))
    label_files = sorted(os.listdir(label_dir))

    # 遍历文件
    for img_file, label_file in zip(img_files, label_files):
        img_path = os.path.join(img_dir, img_file)
        label_path = os.path.join(label_dir, label_file)

        # 加载图像
        image = Image.open(img_path).convert('RGB')
        images.append(image)

        # 加载角度标签
        with open(label_path, 'r') as f:
            angle = float(f.read().strip())
        angles.append(angle)

    return images, angles


if __name__ == '__main__':
    # 加载数据
    data_dir = r"C:\Code\ML\Image\angle_data\train"
    images, angles = load_data(data_dir)

    # model = AngleNet()
    # model.load_state_dict(torch.load('angle_model03.pth'))
    model = torch.load('angle_model04.pt')

    # print(model)
    # 创建数据集和数据加载器
    dataset = AngleDataset(images, angles)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    # 训练epochs
    num_epochs = 40
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            # 前向传播
            outputs = model(inputs)
            labels = labels.to(torch.float).unsqueeze(-1)

            loss = criterion(outputs, labels)

            labels = labels.to(torch.float)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch {epoch + 1} Loss: {epoch_loss:.4f}')

    # 保存模型
    # torch.save(model.state_dict(), 'angle_model03.pth')
    torch.save(model, 'angle_model04.pt')

