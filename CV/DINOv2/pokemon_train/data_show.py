import os
import matplotlib.pyplot as plt
from PIL import Image
import random
import numpy as np
import cv2
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

# ================= 配置区域 =================
IMG_PATH = r"3b4395b45edc29649ae1bdd43596ddb1.png"

# ================= 自定义模拟反光增强 =================
class RandomGlareA(ImageOnlyTransform):
    """Albumentations 版本的模拟卡牌物理反光/高光带"""

    def __init__(self, p=0.5):
        super().__init__(p=p)

    def apply(self, img, **params):
        h, w = img.shape[:2]

        overlay = np.zeros((h, w, 4), dtype=np.uint8)

        x1 = random.randint(0, w // 2)
        y1 = 0
        x2 = random.randint(w // 2, w)
        y2 = h
        width = random.randint(20, max(20, h // 3))
        alpha = random.randint(50, 150)

        cv2.line(overlay, (x1, y1), (x2, y2), (255, 255, 255, alpha), thickness=width)

        img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

        mask = overlay[:, :, 3] / 255.0
        for c in range(3):
            img_rgba[:, :, c] = (1.0 - mask) * img_rgba[:, :, c] + mask * overlay[:, :, c]

        return cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)


# ================= 定义增强 =================
IMG_SIZE = 392
view_transforms = A.Compose([
    # 0. 基础 Resize 保持不变，确保全图都在
    A.Resize(height=IMG_SIZE, width=IMG_SIZE, interpolation=cv2.INTER_CUBIC),

    # 1. 模拟真实透视变形 (幅度调小一点，防止把角拉伸出画面)
    A.Perspective(scale=(0.02, 0.05), p=0.5),

    A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1,
                 num_flare_circles_lower=1, num_flare_circles_upper=3,
                 src_radius=150, src_color=(255, 255, 255), p=0.3),

    # 2. 解决发灰、对比度极低的光照环境 (极其关键，解决小拉达变铁蚁的核心！)
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=(-0.4, 0.1), p=0.6),
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),

    # 3. 模拟漫反射大面积反光白雾 (兼容新旧版本写法)
    # A.RandomSunFlare(src_radius=100, p=0.3),

    # 4. 原有的条状物理反光
    RandomGlareA(p=0.4),

    # 5. 模拟手机镜头模糊和噪点
    A.OneOf([
        A.MotionBlur(blur_limit=5, p=1.0),
        A.GaussianBlur(blur_limit=5, p=1.0),
    ], p=0.3),
    A.GaussNoise(std_range=(0.1, 0.3), p=0.4),

    # 6. 模拟手指遮挡 (把最大遮挡块调小一点，防止刚好把左下角全部捂住)
    A.CoarseDropout(num_holes_range=(2, 4), hole_height_range=(16, 32), hole_width_range=(16, 32), fill=0, p=0.4),

    # 7. YOLO 抠图带来的轻微旋转误差
    A.SafeRotate(limit=5, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5),
])

def visualize_augmentation(img_path):
    if not os.path.exists(img_path):
        print(f"错误: 找不到文件 {img_path}")
        return

    # PIL -> numpy
    original_img = np.array(Image.open(img_path).convert("RGB"))

    # Albumentations 的调用方式
    aug_imgs = [view_transforms(image=original_img)["image"] for _ in range(3)]

    plt.figure(figsize=(16, 6))

    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(original_img)
    plt.axis("off")

    for i, aug_img in enumerate(aug_imgs):
        plt.subplot(1, 4, i + 2)
        plt.title(f"Augmented {i + 1}")
        plt.imshow(aug_img)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    visualize_augmentation(IMG_PATH)