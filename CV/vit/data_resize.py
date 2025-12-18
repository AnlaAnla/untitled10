import os
import shutil
from PIL import Image, PngImagePlugin
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ================= 配置 =================
SOURCE_DIR = r"/home/martin/ML/Image/CardCls/pokemon_tc_us/train"
TARGET_DIR = r"/home/martin/ML/Image/CardCls/pokemon_tc_us/train_resized"
IMG_SIZE = 256  # 建议存成 256，给训练时的增强留一点裁剪空间
# =======================================

# 破解 PIL 限制，防止读取原图时报错
PngImagePlugin.MAX_TEXT_CHUNK = 1000 * (1024 ** 2)
Image.MAX_IMAGE_PIXELS = None


def process_one_class(args):
    """处理单个类别文件夹"""
    class_name, class_src_path, class_dst_path = args

    if not os.path.exists(class_dst_path):
        os.makedirs(class_dst_path)

    files = os.listdir(class_src_path)
    count = 0

    for fname in files:
        src_file = os.path.join(class_src_path, fname)

        # 改名为 .jpg (统一格式，且 JPG 读取速度比 PNG 快)
        name_part = os.path.splitext(fname)[0]
        dst_file = os.path.join(class_dst_path, name_part + ".jpg")

        # 如果已经存在，跳过
        if os.path.exists(dst_file):
            continue

        try:
            with Image.open(src_file) as img:
                # 1. 转换色彩空间 (解决 Palette/Transparency 警告)
                img = img.convert('RGB')

                # 2. 调整大小 (保持比例还是强制拉伸？这里用 Resize 会稍微快一点)
                # 为了特征提取，建议强制拉伸或者保持比例填充黑边
                # 这里为了简单，直接 Resize
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)

                # 3. 保存为 JPG (去掉 PNG 的复杂元数据，彻底解决你的报错)
                img.save(dst_file, "JPEG", quality=90)
                count += 1
        except Exception as e:
            print(f"\n[错误] 无法处理文件 {src_file}: {e}")
            # 坏文件直接丢弃，不要了

    return count


def main():
    if not os.path.exists(SOURCE_DIR):
        print("源目录不存在")
        return

    # 获取所有类别目录
    classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    print(f"找到 {len(classes)} 个类别，准备开始多进程处理...")

    # 准备任务参数
    tasks = []
    for cls in classes:
        src = os.path.join(SOURCE_DIR, cls)
        dst = os.path.join(TARGET_DIR, cls)
        tasks.append((cls, src, dst))

    # 开启多进程加速 (CPU核心数 - 2)
    num_workers = max(1, cpu_count() - 2)

    with Pool(num_workers) as pool:
        # 使用 tqdm 显示进度
        results = list(tqdm(pool.imap(process_one_class, tasks), total=len(tasks), unit="class"))

    total_images = sum(results)
    print(f"\n处理完成！共生成 {total_images} 张图片。")
    print(f"新数据集位置: {TARGET_DIR}")
    print("请修改训练脚本中的 TRAIN_DIR 指向新位置。")


if __name__ == '__main__':
    main()
