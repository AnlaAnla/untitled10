from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import platform

cudnn.benchmark = True
plt.ion()  # interactive mode


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
    # 计算完成百分比
    percent_complete = f"{(100 * (iteration / float(total))):.{decimals}f}"
    # 计算进度条填充长度
    filled_length = int(length * iteration // total)
    # 创建进度条字符串
    bar = fill * filled_length + '-' * (length - filled_length)
    # 打印进度条
    print(f'\r{prefix} |{bar}| {percent_complete}% {suffix}', end=print_end)
    # 完成时打印新行
    if iteration == total:
        print()


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in data_phase:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            l = len(dataloaders[phase])
            print_progress_bar(0, l, prefix='进度:', suffix='完成', length=50)

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.chunk)

                # 更新进度条
                print_progress_bar(i + 1, l, prefix='进度:', suffix='完成', length=50)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.6f} Acc: {epoch_acc:.6f}')

            # deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('copy best model')

        if (epoch + 1) % 10 == 0:
            print("save temp model in ", epoch + 1)
            torch.save(model.state_dict(), 'card_resnet_temp.pth')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # 加载在验证集中表现最好的模型
    model.load_state_dict(best_model_wts)

    return model


def train(epoch, out_features, save_path='mobilenetv3.pth', load_my_model=False, model_path=None,
          is_freeze=False, freeze_num=16):

    # 如果不加载训练过的模型则加载预训练模型
    if load_my_model:
        model = models.mobilenet_v3_large(pretrained=False)
        # 修改最后一层
        model.classifier[-1] = torch.nn.Linear(in_features=1280, out_features=out_features, bias=True)
        model.load_state_dict(torch.load(model_path))

    else:
        model = models.mobilenet_v3_large(pretrained=True)
        # 修改最后一层
        model.classifier[-1] = torch.nn.Linear(in_features=1280, out_features=out_features, bias=True)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # 冻结部分参数
    if is_freeze:
        for i, c in enumerate(model.features.children()):
            if i == freeze_num:
                break
            for param in c.parameters():
                param.requires_grad = False

        for param in model.features.parameters():
            print(param.requires_grad)

    model = train_model(model, criterion, optimizer_ft,
                        exp_lr_scheduler, num_epochs=epoch)

    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    data_dir = r"D:\Code\ML\images\Mywork3\train_data"
    # model_path = "D:/Code/ML/model/card_cls/effcient_card_out1945_freeze6.pth"

    # if platform.system() == 'Windows':
    #     print('这是Windows系统')
    #     # data_dir = os.path.join('D:', data_dir)
    #     # model_path = os.path.join('D:', model_path)
    # elif platform.system() == 'Linux':
    #     print('这是Linux系统')
    #     data_dir = os.path.join("/mnt/d", data_dir)
    #     model_path = os.path.join("/mnt/d", model_path)
    # else:
    #     print(platform.system())

    print('dataset: ', data_dir)
    # print('model_path', model_path)

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),

            transforms.RandAugment(),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomApply([transforms.GaussianBlur(5)], p=0.3),

            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_phase = ['train', 'val']
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in data_phase}
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=16, pin_memory=True)
        for x in data_phase}
    dataset_sizes = {x: len(image_datasets[x]) for x in data_phase}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class_num = len(class_names)
    print('class:', class_num)

    '''
    epoch: 训练次数
    save_path: 模型保存路径
    load_my_model: 是否加载训练过的模型
    model_path: 加载的模型路径 
    is_freeze: 是否冻结模型部分层数
    freeze_num: 冻结层数
    '''
    data_phase = ['train', 'val']
    # 数据集路径在本文件上面

    # 初次训练
    train(epoch=30, out_features=class_num,
          save_path='mobilenetv3_out1945_card01.pth',
          load_my_model=False, model_path=None,
          is_freeze=True)
