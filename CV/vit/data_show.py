import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# ================= 配置区域 =================
# 替换为你想要测试的图片路径
IMG_PATH = r"C:\Code\ML\Image\_CLASSIFY\card_cls2\Pokemon\伊布_us\1ce1a638-5754-45c2-81d0-b46fb2ba7671.png"

# ================= 定义增强 (去掉了缩放和归一化) =================
# 注意：RandomPerspective 和 GaussianBlur 现在的 torchvision 版本支持直接对 PIL 图像操作
view_transforms = transforms.Compose([
    # ---------------- 被移除的缩放层 ----------------
    # transforms.Resize((240, 240)),
    # transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),

    # ---------------- 保留的增强层 ----------------
    # 透视变换 (模拟拍摄角度倾斜)
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),

    # 随机旋转 (模拟卡片没摆正)
    transforms.RandomRotation(degrees=20),

    # 颜色抖动 (模拟光照、色差)
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),

    # 高斯模糊 (模拟对焦不准)
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),

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