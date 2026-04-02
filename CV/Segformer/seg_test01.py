import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import matplotlib.pyplot as plt
import requests

# 1. 准备图片 (你可以替换为本地图片的路径 image = Image.open("your_image.jpg"))
img_path = r"C:\Users\wow38\Pictures\videoframe_6871967.png"
image = Image.open(img_path).convert("RGB")

# 2. 加载图像处理器和模型
model_name = r"C:\Code\ML\Model\Card_Seg\segformer_card_hand01.pt"
print("正在加载模型...")
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForSemanticSegmentation.from_pretrained(model_name)

# 3. 预处理图像
inputs = processor(images=image, return_tensors="pt")

# 4. 进行推理 (Forward pass)
with torch.no_grad():
    outputs = model(**inputs)

# 5. 后处理输出结果
# 模型输出的 logits 尺寸通常是原图的 1/4，需要插值（上采样）回原始图像的尺寸
logits = outputs.logits
upsampled_logits = torch.nn.functional.interpolate(
    logits,
    size=image.size[::-1], # 目标尺寸：(height, width)
    mode="bilinear",
    align_corners=False,
)

# 获取每个像素点预测的类别索引 (argmax)
predicted_segmentation_map = upsampled_logits.argmax(dim=1)[0]
segmentation_map_np = predicted_segmentation_map.cpu().numpy()

# ---- 附加：打印图像中识别出的物体类别 ----
print("\n这张图像中检测到的类别有:")
detected_labels = torch.unique(predicted_segmentation_map).tolist()
for label_id in detected_labels:
    class_name = model.config.id2label[label_id]
    print(f"- {class_name}")

# 6. 结果可视化
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# 显示原图
axs[0].imshow(image)
axs[0].set_title("Original Image")
axs[0].axis("off")

# 显示分割图叠加在原图上
axs[1].imshow(image)
# 使用 cmap="jet" 或 "tab20" 生成彩色掩码，alpha控制透明度
axs[1].imshow(segmentation_map_np, cmap="tab20", alpha=0.6)
axs[1].set_title("Segmentation Mask Overlay")
axs[1].axis("off")

plt.tight_layout()
plt.show()
