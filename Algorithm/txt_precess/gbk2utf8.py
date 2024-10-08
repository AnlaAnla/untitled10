import os
import chardet

# 指定目录路径
dir_path = r'C:\Code\ML\Image\angle_data\test\label'

# 遍历目录下所有文件
for filename in os.listdir(dir_path):
    # 只处理txt文件
    if filename.endswith('.txt'):
        file_path = os.path.join(dir_path, filename)

        # 检测文件编码
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']

        # 如果不是UTF-8编码,则转换为UTF-8
        if encoding != 'utf-8':
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f'{filename} 已转换为UTF-8编码')