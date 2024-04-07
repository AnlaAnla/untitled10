import torch
import timm
from PIL import Image
from transformers import ViTForImageClassification
from torchvision import transforms, datasets
import requests
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import os
import numpy as np
import torch.backends.cudnn as cudnn
import time
import copy

cudnn.benchmark = True


norm_mean = [0.5, 0.5, 0.5]
norm_std = [0.5, 0.5, 0.5]
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandAugment(),
        # transforms.Resize(384),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ]),
}

data_phase = ['train', 'val']
# data_phase = ['train']
data_dir = r"D:\Code\ML\images\Mywork3\train_data3"

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in data_phase}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=8)
               for x in data_phase}
dataset_sizes = {x: len(image_datasets[x]) for x in data_phase}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 定义一个函数，用来打印动态进度条
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


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in data_phase:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            l = len(dataloaders[phase])
            print_progress_bar(0, l, prefix='进度:', suffix='完成', length=50)

            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).pooler_output
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # 更新进度条
                print_progress_bar(i + 1, l, prefix='进度:', suffix='完成', length=50)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0:
            print("save temp model in ", epoch + 1)
            torch.save(model.state_dict(), 'card_resnet_temp.pth')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


if __name__ == '__main__':

    print('class:', len(class_names))

    model = torch.load('vit-large-patch32-384.pt')
    model.pooler = torch.nn.Sequential(
        torch.nn.Linear(in_features=2560, out_features=len(class_names), bias=False),
        torch.nn.Tanh()
    )
    # model.head = torch.nn.Linear(model.head.in_features, 854)

    # model.load_state_dict(torch.load(r"D:\Code\ML\model\card_cls\vit_card_out854_freeze1.pth"))

    # model.head = torch.nn.Linear(model.head.in_features, len(class_names))
    # 冻结部分参数
    # for param in model_conv.parameters():
    #     param.requires_grad = False

    # 冻结部分参数
    # 冻结除了最后一层以外的参数
    for i, c in enumerate(model.children()):
        if i >= 2:
            break
        for param in c.parameters():
            param.requires_grad = False
    for name, param in model.named_parameters():
        print(name, ': ', param.requires_grad)
    # Parameters of newly constructed modules have requires_grad=True by default

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model = train_model(model, criterion, optimizer_ft,
                        exp_lr_scheduler, num_epochs=5)

    torch.save(model.state_dict(), 'vit_large_card_out854_freeze2.pth')

# print(model)
# model_name_or_path = 'google/vit-base-patch16-224'
# model = ViTForImageClassification.from_pretrained(
#     model_name_or_path,
# )

# norm_mean = [0.5, 0.5, 0.5]
# norm_std = [0.5, 0.5, 0.5]
# inference_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(norm_mean, norm_std),
# ])

# image_url = 'https://i1.hdslb.com/bfs/archive/244d0c76d5d2f0f77d43bfcadfe8d0289b63ae24.jpg@672w_378h_1c_!web-home-common-cover'
# image = Image.open(requests.get(image_url, stream=True).raw)
# input_tensor = inference_transform(image).unsqueeze(0)
#
# with torch.no_grad():
#     outputs00 = model(input_tensor)
#     predictions = torch.argmax(outputs00, dim=1)
#
# print(predictions)
# print("1, Predicted class:", model.config.id2label[predictions.item()])
