conda activate pytorch  # 激活 conda pytorch 环境
python test01.py  # 执行 stream01.py 脚本
python test02.py  # 执行 test02.py 脚本

# 检查当前目录是否存在 A 文件夹
if [ -d "CV" ]; then
    python test03.py  # 如果存在 A 文件夹，执行 test03.py
else
    python test04.py  # 如果不存在 A 文件夹，执行 test04.py
fi