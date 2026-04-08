import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

# ================= 配置区域 =================
# 替换为你想要测试的图片路径
IMG_PATH = r"C:\Code\ML\Project\PokemonCardSearch\temp_yolo.jpg"

class PadToSquare:
    def __init__(self, fill=(128, 128, 128)):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        max_wh = max(w, h)
        pad_w = max_wh - w
        pad_h = max_wh - h
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        return TF.pad(img, padding, fill=self.fill, padding_mode='constant')


class RandomBackgroundPad:
    """以一定概率将卡牌稍微缩小，并放置在随机颜色的纯色背景上，模拟拍照时拍到了桌面的情况"""

    def __init__(self, p=0.5, scale_range=(0.85, 0.95)):
        self.p = p
        self.scale_range = scale_range

    def __call__(self, img):
        if random.random() > self.p:
            return img

        w, h = img.size
        scale = random.uniform(*self.scale_range)
        new_w, new_h = int(w * scale), int(h * scale)

        # 缩小图像
        img_resized = TF.resize(img, (new_h, new_w))

        # 生成随机背景色 (模拟各种颜色的桌面)
        bg_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # 补回原尺寸
        pad_w = w - new_w
        pad_h = h - new_h
        # 随机位置放置 (不一定在正中间，模拟没拍正)
        left = random.randint(0, pad_w)
        top = random.randint(0, pad_h)
        padding = (left, top, pad_w - left, pad_h - top)

        return TF.pad(img_resized, padding, fill=bg_color, padding_mode='constant')


# ================= 定义增强 (去掉了缩放和归一化) =================
# 注意：RandomPerspective 和 GaussianBlur 现在的 torchvision 版本支持直接对 PIL 图像操作
view_transforms = transforms.Compose([
    # ---------------- 被移除的缩放层 ----------------
    RandomBackgroundPad(p=0.4, scale_range=(0.85, 0.95)),
    PadToSquare(fill=(128, 128, 128)),

    transforms.Resize((224, 224)),

    # ---------------- 保留的增强层 ----------------
    # degrees: 微微旋转 | translate: 平移 | scale: 缩放 | shear: 错切(模拟轻微倾斜)
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=3),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
    transforms.ColorJitter(brightness=(0.8, 1.3), contrast=(0.7, 1.1)),

    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),

    # ---------------- 被移除的张量化和归一化 ----------------
    # transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def visualize_augmentation(img_path):
    if not os.path.exists(img_path):
        print(f"错误: 找不到文件 {img_path}")
        return

    # 1. 读取原图
    original_img = Image.open(img_path).convert("RGB")

    # 2. 生成 3 张随机增强后的图片
    aug_imgs = [view_transforms(original_img) for _ in range(3)]

    # 3. 绘图
    plt.figure(figsize=(16, 6))

    # 显示原图
    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(original_img)
    plt.axis('off')

    # 显示增强图
    for i, aug_img in enumerate(aug_imgs):
        plt.subplot(1, 4, i + 2)
        plt.title(f"Augmented {i + 1}")
        plt.imshow(aug_img)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 如果你是远程运行，看不到 plt.show() 的窗口
    # 你可以将图片保存下来查看，或者使用 Jupyter Notebook
    visualize_augmentation(IMG_PATH)

    # 如果在远程终端无法弹窗，可以使用下面的代码保存结果：
    # plt.savefig("aug_result.png")
    # print("结果已保存为 aug_result.png")
