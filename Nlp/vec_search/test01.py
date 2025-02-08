import numpy as np

# 加载向量库和名称库
vec_data = np.load("temp/vec_data_text01.npy")
name_list = np.load("temp/vec_data_text01_names.npy")
print('加载向量库和名称库')

# 检查第一个向量是否为零向量
if np.all(vec_data[0] == 0):
    print("移除第一个零向量...")
    # 移除第一个向量
    vec_data = vec_data[1:]
    # 移除对应的名称
    name_list = name_list[1:]

    # 保存修改后的向量库和名称库
    np.save("temp/vec_data_text01.npy", vec_data)
    np.save("temp/vec_data_text01_names.npy", name_list)
    print("已移除零向量并更新文件")
else:
    print("第一个向量不是零向量，无需移除")

print("vec_data shape:", vec_data.shape)
print("name_list length:", len(name_list))