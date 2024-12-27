import subprocess
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 要执行的命令
cmd = "ct2-transformers-converter --model mbart-large-50-many-to-many-mmt --output_dir mbart-large-50-many-to-many-mmt-ct2"

# 执行命令
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

# 检查命令是否成功执行
if result.returncode == 0:
    print("命令执行成功。")
    # 打印命令输出
    print(f"标准输出:\n{result.stdout}")
else:
    print("命令执行失败。")
    # 打印错误信息
    print(f"标准错误:\n{result.stderr}")