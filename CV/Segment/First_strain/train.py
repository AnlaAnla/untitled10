import torch
import torchvision
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
model.eval()

# print(model)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


input_image = Image.open(r"C:\Code\ML\Image\_TEST_DATA\Card_test\test03\11.jpg").convert('RGB')
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

with torch.no_grad():
    output = model(input_batch)['out'][0]

output_predictions = output.argmax(0)


# create a color pallette, selecting a color for each class
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# plot the semantic segmentation predictions of 21 classes in each color
r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
r.putpalette(colors)


plt.imshow(r)
plt.show()

print()