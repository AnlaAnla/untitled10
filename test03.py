import json

path = r"C:\Code\ML\Image\yolo_data02\Card_scratch\Scratch\scratch1\anylabels\image - 2024-07-24T114533.996.json"
with open(path, encoding='utf-8') as f:
    data = json.load(f)
print(data)

print("验证")