import os

def setting():
    # 设置pytorch缓存地址
    cache_path = r"D:\Code\ML\Pretrain_model\torch_model"
    os.environ['TORCH_HOME'] = cache_path