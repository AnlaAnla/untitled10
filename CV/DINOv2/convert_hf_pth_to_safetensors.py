import torch
import torch.nn.functional as F
import os
import math
from safetensors.torch import save_file
from transformers import Dinov2Config

# ================= 配置区域 =================

# 1. 输入: 你的训练结果
# INPUT_PATH = r"/home/martin/ML/Model/card_retrieval/dinov2_base_392_CardSetTextIcon01/best_model.pth"
INPUT_PATH = r"/home/martin/ML/Model/pokemon_cls/dinov2_contrastive_392/best_retrieval_model.pth"

# 2. 输出文件夹
# OUTPUT_DIR = r"/home/martin/ML/Model/card_cls/dinov2_base_392_CardSetTextIcon01"
OUTPUT_DIR = r"/home/martin/ML/Model/card_cls/dinov2_base_retrieval_392_PokemonCN04"

# 3. 图像尺寸 (必须和训练时一致)
IMG_SIZE = 392
PATCH_SIZE = 14


# ===========================================

def interpolate_pos_encoding(pos_embed, target_img_size, patch_size):
    """
    将位置编码从原始尺寸插值到目标尺寸
    Args:
        pos_embed: [1, N, Dim] 例如 [1, 1370, 768]
        target_img_size: int 例如 392
        patch_size: int 例如 14
    """
    N = pos_embed.shape[1] - 1  # 减去 CLS token
    dim = pos_embed.shape[2]

    # 计算原始的网格大小 (sqrt(1369) = 37)
    orig_size = int(math.sqrt(N))

    # 计算目标的网格大小 (392 / 14 = 28)
    new_size = target_img_size // patch_size

    if orig_size == new_size:
        return pos_embed

    print(f"⚠️ 检测到尺寸变化: 原网格 {orig_size}x{orig_size} -> 新网格 {new_size}x{new_size}")

    # 分离 CLS token 和 Patch tokens
    class_pos_embed = pos_embed[:, 0]  # [1, 768]
    patch_pos_embed = pos_embed[:, 1:]  # [1, 1369, 768]

    # 维度变换: [B, N, C] -> [B, C, H, W] 以便使用 interpolate
    # [1, 1369, 768] -> [1, 768, 37, 37]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 1).reshape(1, dim, orig_size, orig_size)

    # 双线性插值
    patch_pos_embed = F.interpolate(
        patch_pos_embed,
        size=(new_size, new_size),
        mode='bicubic',
        align_corners=False,
    )

    # 变换回: [B, C, H, W] -> [B, N_new, C]
    # [1, 768, 28, 28] -> [1, 784, 768]
    patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)

    # 重新拼接 CLS token
    # [1, 1, 768] + [1, 784, 768] -> [1, 785, 768]
    new_pos_embed = torch.cat((class_pos_embed.unsqueeze(1), patch_pos_embed), dim=1)

    return new_pos_embed


def main():
    print(f"📂 正在加载训练权重: {INPUT_PATH}")
    if not os.path.exists(INPUT_PATH):
        print("❌ 文件不存在")
        return

    # 加载权重
    checkpoint = torch.load(INPUT_PATH, map_location="cpu")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    new_dict = {}
    bn_neck_count = 0
    backbone_count = 0

    print("🔄 正在清洗 Key 名称并处理位置编码...")

    # 查找 embeddings.position_embeddings 的 key 名
    # 通常是 backbone.embeddings.position_embeddings 或者 embeddings.position_embeddings

    for key, value in state_dict.items():
        new_key = None

        # Key 清洗逻辑
        if key.startswith("backbone."):
            new_key = key.replace("backbone.", "")
        elif key.startswith("bn_neck."):
            new_key = key
        elif key.startswith("module.backbone."):
            new_key = key.replace("module.backbone.", "")
        elif key.startswith("module.bn_neck."):
            new_key = key.replace("module.", "")

        if "head." in key:
            continue

        if new_key:
            # === 关键修复步骤: 检查并处理位置编码 ===
            if "embeddings.position_embeddings" in new_key:
                print(f"📏 正在调整位置编码: {new_key}，原始形状: {value.shape}")
                # 调用插值函数
                value = interpolate_pos_encoding(value, IMG_SIZE, PATCH_SIZE)
                print(f"   -> 调整后形状: {value.shape}")
            # ========================================

            new_dict[new_key] = value

            if "bn_neck" in new_key:
                bn_neck_count += 1
            else:
                backbone_count += 1

    if backbone_count == 0:
        print("⚠️ 警告: 没有找到 backbone 权重！")
        return

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === 保存 model.safetensors ===
    save_path = os.path.join(OUTPUT_DIR, "model.safetensors")
    save_file(new_dict, save_path)
    print(f"💾 权重已保存: {save_path}")

    # === 保存 config.json ===
    print("📝 生成 Config 文件...")
    config = Dinov2Config(
        image_size=IMG_SIZE,  # 392
        patch_size=PATCH_SIZE,  # 14
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        mlp_ratio=4.0,
        architectures=["Dinov2Model"]
    )
    config.save_pretrained(OUTPUT_DIR)
    print(f"✅ 转换全部完成! 输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()