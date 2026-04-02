import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

# ===== 你自己的路径 =====
base_model = "nvidia/segformer-b0-finetuned-ade-512-512"
ckpt_path = r"C:\Code\ML\Model\Card_Seg\segformer_card_hand02.pt"
save_dir = r"C:\Code\ML\Model\Card_Seg\segformer_card_hand02_safetensors"

# 你的类别，要和训练时一致
class_names = ["background", "card", "hand"]
num_classes = len(class_names)

id2label = {i: name for i, name in enumerate(class_names)}
label2id = {name: i for i, name in enumerate(class_names)}

print("1) 加载基础 processor ...")
processor = AutoImageProcessor.from_pretrained(base_model)

print("2) 构建模型结构 ...")
model = AutoModelForSemanticSegmentation.from_pretrained(
    base_model,
    num_labels=num_classes,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

print("3) 读取你的 checkpoint ...")
checkpoint = torch.load(ckpt_path, map_location="cpu")

if "model" not in checkpoint:
    raise ValueError("这个 .pt 文件里没有 'model' 键，说明它不是你训练脚本保存的 checkpoint 格式。")

state_dict = checkpoint["model"]

print("4) 加载权重 ...")
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
print("missing_keys:", missing_keys)
print("unexpected_keys:", unexpected_keys)

print("5) 导出为 safetensors 格式 ...")
model.save_pretrained(save_dir, safe_serialization=True)
processor.save_pretrained(save_dir)

print("转换完成，输出目录：", save_dir)