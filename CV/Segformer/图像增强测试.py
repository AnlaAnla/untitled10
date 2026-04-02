import argparse
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from PIL import Image

from train_segformer import SegTransform


# ---------------------------------
# 图像增强可视化小工具
# ---------------------------------


def setup_chinese_font() -> str | None:
    # 尝试设置一个可用的中文字体，避免 matplotlib 的中文乱码/缺字警告
    preferred = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return None


def parse_args():
    img_path = r"C:\Users\wow38\Pictures\videoframe_6871967.png"

    parser = argparse.ArgumentParser(description="Visualize SegFormer train-time augmentations")
    parser.add_argument("--img", type=str, default=img_path, help="输入图片路径")
    parser.add_argument("--num", type=int, default=6, help="要展示的增强样本数量")
    parser.add_argument("--img-size", type=int, default=512, help="训练时的目标尺寸")

    # 下面这些参数与 train_segformer.py 中的增强配置对应
    parser.add_argument("--scale", type=float, nargs=2, default=(0.8, 1.2), help="随机缩放范围")
    parser.add_argument("--hflip", type=float, default=0.5, help="水平翻转概率")
    parser.add_argument("--vflip", type=float, default=0.0, help="垂直翻转概率")
    parser.add_argument("--color-jitter", type=float, nargs=3, default=(0.2, 0.2, 0.2), help="亮度/对比度/饱和度扰动")

    parser.add_argument("--seed", type=int, default=42, help="随机种子，方便复现")
    parser.add_argument("--cols", type=int, default=3, help="展示网格的列数")
    return parser.parse_args()


def main():
    args = parse_args()

    # 固定随机种子（可复现同一组增强效果）
    random.seed(args.seed)

    # 设置中文字体（如果系统里有）
    font_name = setup_chinese_font()
    if font_name is None:
        print("未找到可用中文字体，标题可能出现缺字提示。")

    img_path = Path(args.img)
    if not img_path.exists():
        raise FileNotFoundError(f"找不到图片: {img_path}")

    # 读取原图并转成 RGB，防止通道格式不一致
    image = Image.open(img_path).convert("RGB")

    # 生成一个假 mask（全 0），只为了复用同一套增强逻辑
    dummy_mask = Image.new("L", image.size, 0)

    # 初始化与训练相同的增强器
    augmenter = SegTransform(
        img_size=args.img_size,
        train=True,
        scale_min=args.scale[0],
        scale_max=args.scale[1],
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=tuple(args.color_jitter),
    )

    # 准备要显示的图片列表：第一张是原图（仅 resize 以对齐尺寸）
    images = []
    original = image.resize((args.img_size, args.img_size), Image.BILINEAR)
    images.append(("原图", original))

    # 生成多份增强后的图像
    for i in range(args.num):
        aug_img, _ = augmenter(image, dummy_mask)
        images.append((f"增强 {i + 1}", aug_img))

    # 计算网格尺寸
    total = len(images)
    cols = max(1, args.cols)
    rows = math.ceil(total / cols)

    # 用 matplotlib 可视化
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for ax, (title, img) in zip(axes, images):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # 多余的子图隐藏
    for ax in axes[len(images):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
