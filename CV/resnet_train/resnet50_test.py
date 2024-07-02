import torch
from torchvision import transforms
from PIL import Image
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

classes = []


def inference_transform():
    inference_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return inference_transform


def get_img_tensor(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = inference_transform()
    img_tensor = transform(img)
    img_tensor.unsqueeze_(0)

    return img_tensor.to(device)

model = torch.load('res_card.pt')
model.to(device)
model.eval()


if __name__ == '__main__':
    dir_path = r"D:\Code\ML\model\card_cls\res_card_freeze.pth"
    for name in os.listdir(dir_path):
        img_path = os.path.join(dir_path, name)
        img_name = os.path.split(img_path)[-1].split('.')[0]

        img_tensor = get_img_tensor(img_path)
        result = model(img_tensor)
        print('name:',img_name, "  --  pre:", classes[result.argmax()])

