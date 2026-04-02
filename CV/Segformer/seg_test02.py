import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

model_dir = r"C:\Code\ML\Model\Card_Seg\segformer_card_hand02_safetensors"
# img_path = r"C:\Users\wow38\Pictures\videoframe_6871967.png"



processor = AutoImageProcessor.from_pretrained(model_dir)
model = AutoModelForSemanticSegmentation.from_pretrained(model_dir)

def show(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    pred = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False
    ).argmax(dim=1)[0].cpu().numpy()

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(pred)
    plt.title("mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    show(r"C:\Code\ML\Project\CardVideoSummary\static\frames\7eb64157-3ad6-4014-a9cb-61e7b708f790_5754600.jpg")
    print()
