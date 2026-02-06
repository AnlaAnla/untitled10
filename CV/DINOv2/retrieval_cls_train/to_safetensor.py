import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from transformers import Dinov2Config, Dinov2Model
from safetensors.torch import save_file


def interpolate_pos_encoding(pos_embed, target_len):
    """
    对位置编码进行插值调整
    pos_embed: [1, N_original, Dim]
    target_len: N_target (例如 785)
    """
    if pos_embed.shape[1] == target_len:
        return pos_embed

    print(f"⚠️ 检测到位置编码尺寸不匹配: 权重 {pos_embed.shape} -> 目标 {target_len}")
    print("   正在执行双三次插值 (Bicubic Interpolation)...")

    # 1. 分离 CLS token 和 Patch tokens
    cls_token = pos_embed[:, 0:1, :]
    patch_tokens = pos_embed[:, 1:, :]

    dim = pos_embed.shape[-1]

    # 2. 计算原始网格大小和目标网格大小
    # 原来的 patch 数量
    n_patches_origin = patch_tokens.shape[1]
    w0 = h0 = int(math.sqrt(n_patches_origin))

    # 目标的 patch 数量 (target_len - 1)
    n_patches_target = target_len - 1
    w_new = h_new = int(math.sqrt(n_patches_target))

    print(f"   Grid 变换: {w0}x{h0} -> {w_new}x{h_new}")

    # 3. 变换形状以进行插值: [1, N, C] -> [1, C, H, W]
    patch_tokens = patch_tokens.reshape(1, w0, h0, dim).permute(0, 3, 1, 2)

    # 4. 执行插值
    patch_tokens = F.interpolate(
        patch_tokens,
        size=(w_new, h_new),
        mode='bicubic',
        align_corners=False
    )

    # 5. 变回形状: [1, C, H, W] -> [1, N_new, C]
    patch_tokens = patch_tokens.permute(0, 2, 3, 1).reshape(1, -1, dim)

    # 6. 拼回 CLS token
    new_pos_embed = torch.cat((cls_token, patch_tokens), dim=1)

    return new_pos_embed


def convert_dino_to_hf(pth_path, output_dir):
    print(f"🔄 正在处理: {pth_path}")

    checkpoint = torch.load(pth_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    # === 配置区域 ===
    # 目标是 392，对应 785 个 token
    config = Dinov2Config(
        image_size=392,
        patch_size=14,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        architectures=["Dinov2Model"]
    )

    hf_model = Dinov2Model(config)

    # 映射字典
    new_state_dict = {}
    bn_neck_weights = {}

    for key, value in state_dict.items():
        key = key.replace('module.', '')

        # 保存 BN Neck
        if key.startswith('bn_neck.'):
            bn_neck_weights[key] = value
            continue

        if not key.startswith('backbone.'):
            continue

        old_key = key.replace('backbone.', '')
        new_key = old_key

        # 1. Embeddings
        if 'cls_token' in old_key:
            new_key = 'embeddings.cls_token'
        elif 'mask_token' in old_key:
            new_key = 'embeddings.mask_token'
        elif 'pos_embed' in old_key:
            new_key = 'embeddings.position_embeddings'
            # === 这里调用插值函数 ===
            # 目标长度是 config 中的 expected seq length
            target_seq_len = (config.image_size // config.patch_size) ** 2 + 1  # 785
            value = interpolate_pos_encoding(value, target_seq_len)

        elif 'patch_embed.proj' in old_key:
            new_key = old_key.replace('patch_embed.proj', 'embeddings.patch_embeddings.projection')

        # 2. Blocks
        elif 'blocks.' in old_key:
            parts = old_key.split('.')
            layer_idx = parts[1]
            rest = ".".join(parts[2:])
            base = f"encoder.layer.{layer_idx}"

            if 'norm1' in rest:
                new_key = f"{base}.norm1.{rest.split('.')[-1]}"
            elif 'norm2' in rest:
                new_key = f"{base}.norm2.{rest.split('.')[-1]}"
            elif 'ls1' in rest:
                new_key = f"{base}.layer_scale1.lambda1"
            elif 'ls2' in rest:
                new_key = f"{base}.layer_scale2.lambda1"
            elif 'mlp.fc1' in rest:
                new_key = f"{base}.mlp.fc1.{rest.split('.')[-1]}"
            elif 'mlp.fc2' in rest:
                new_key = f"{base}.mlp.fc2.{rest.split('.')[-1]}"
            elif 'attn' in rest:
                if 'qkv.weight' in rest:
                    q, k, v = value.chunk(3, dim=0)
                    new_state_dict[f"{base}.attention.attention.query.weight"] = q
                    new_state_dict[f"{base}.attention.attention.key.weight"] = k
                    new_state_dict[f"{base}.attention.attention.value.weight"] = v
                    continue
                elif 'qkv.bias' in rest:
                    q, k, v = value.chunk(3, dim=0)
                    new_state_dict[f"{base}.attention.attention.query.bias"] = q
                    new_state_dict[f"{base}.attention.attention.key.bias"] = k
                    new_state_dict[f"{base}.attention.attention.value.bias"] = v
                    continue
                elif 'proj' in rest:
                    new_key = f"{base}.attention.output.dense.{rest.split('.')[-1]}"

        # 3. Final Norm
        elif key.endswith('norm.weight'):
            new_key = 'layernorm.weight'
        elif key.endswith('norm.bias'):
            new_key = 'layernorm.bias'

        new_state_dict[new_key] = value

    print(f"📊 映射完成: {len(new_state_dict)} keys mapped.")

    # 验证加载 (strict=False 允许缺少 mask_token)
    missing, unexpected = hf_model.load_state_dict(new_state_dict, strict=False)

    # 只要不是 shape mismatch，missing keys 通常是可以接受的 (如 mask_token)
    if unexpected:
        print(f"❌ 出现意外的 Key: {unexpected}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config.save_pretrained(output_dir)

    final_dict = {**new_state_dict, **bn_neck_weights}
    save_file(final_dict, os.path.join(output_dir, "model.safetensors"))

    print(f"✅ 修复并转换成功! 392x392 模型已保存至: {output_dir}")


if __name__ == "__main__":
    BEST_MODEL_PATH = "best_model.pth"
    OUTPUT_DIR = "dinov2_hf_local"

    convert_dino_to_hf(BEST_MODEL_PATH, OUTPUT_DIR)