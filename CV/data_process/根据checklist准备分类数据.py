import pandas as pd
import os
import shutil
import tqdm

data_path = "D:\\Code\\ML\\Embedding\\img_vec\\checklist_ebay_data2023.xlsx"  # 替换为你的 Excel 文件路径
image_dir = "D:\\Code\\ML\\Embedding\\img_vec\\image_yolo224"

checklist_classes_dir = r"D:\Code\ML\Embedding\img_vec\checklist2023_classes\train"

df = pd.read_excel(data_path)

i = 0
for index, row in df.iterrows():
    checklist_id = row['checklist_id']

    img_url = row['img_origin']
    file_name_base = img_url.split("/")[-2]
    file_name = file_name_base + ".jpg"
    file_path = os.path.join(image_dir, file_name)

    if os.path.exists(file_path):
        save_dir = os.path.join(checklist_classes_dir, str(checklist_id))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            img_save_path = os.path.join(save_dir, file_name)
            shutil.copy(file_path, img_save_path)
        else:
            print(f"存在: {checklist_id}, {file_path}")

    i += 1
    if i % 1000 == 0:
        print(i)
