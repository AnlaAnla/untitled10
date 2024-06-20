from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import numpy as np

processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")


def show(img_path):
    image = Image.open(img_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0]
    mask = pred_seg.numpy()
    mask = (mask >= 4) & (mask <= 7)

    # 将mask扩展到与图片相同的形状
    mask_expanded = np.stack([mask] * 3, axis=-1)

    img = np.array(image)
    # 对应mask为False的位置,用白色像素替换
    # 将mask为False的部分变成白色
    img[~mask_expanded] = 255

    plt.imshow(img)
    plt.show()


show(r"C:\Code\ML\Image\Card_test\test\ttt11.jpg")
print()
