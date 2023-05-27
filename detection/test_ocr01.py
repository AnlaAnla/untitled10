import easyocr
import os
import PIL.Image as Image

# reader = easyocr.Reader(['ch_sim','en'])
import numpy as np

allow_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
              'V', 'W', 'X', 'Y', 'Z',
              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
              'v', 'w', 'x', 'y', 'z',
              "'", "\"", ',', '?', '.', ' ']

reader = easyocr.Reader(['en'])
img_path = r"D:\Pictures\2b805ee56776a76cf11aa07489b84a3.jpg"

# result = reader.readtext(os.path.join(dir_path, dir_path), detail=0)
# img = Image.open(dir_path)

text = reader.readtext(img_path, detail=0, batch_size=4, allowlist=allow_list,
                       rotation_info=[-30, 30])
print(text)
# images = os.listdir(dir_path)
# result_list = []
#
# for img_name in images[:20]:
#     result = reader.readtext(os.path.join(dir_path, img_name), detail=0)
#     result_list.append(result)
#     print(img_name)
#
# print(result_list)


