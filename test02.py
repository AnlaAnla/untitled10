import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"可见 GPU 数量: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print("-" * 30)
    for i in range(torch.cuda.device_count()):
        # 这里的显存总量应该是 16GB (V100)，而不是 3GB (1060)
        p = torch.cuda.get_device_properties(i)
        print(f"逻辑设备 cuda:{i} -> {p.name} (显存: {p.total_memory / 1024**3:.1f} GB)")