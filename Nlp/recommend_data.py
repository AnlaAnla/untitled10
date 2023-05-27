import pandas as pd
import numpy as np
import os
import time


# generate user data

# data_dir01 = r"D:\Code\ML\texts\用户点击明细\组队商品浏览"
# data_dir02 = r"D:\Code\ML\texts\用户点击明细\组队商品购买"
#
#
# user_list = []
# for file_name in os.listdir(data_dir01):
#     t1 = time.time()
#
#     file_path = os.path.join(data_dir01, file_name)
#     dataframe = pd.read_excel(file_path)
#     user_list.append(list(set(dataframe['用户编号'])))
#
#     t2 = time.time()
#     print(t2 - t1)

# for file_name in os.listdir(data_dir02):
#     t1 = time.time()
#
#     file_path = os.path.join(data_dir02, file_name)
#     dataframe = pd.read_excel(file_path)
#     user_list.append(list(set(dataframe['用户编号'])))
#
#     t2 = time.time()
#     print(t2 - t1)
#
# merged_list = [item for sublist in user_list for item in sublist]
# user_list = sorted(list(set(merged_list)))
#
# with open('user_buy.data', 'w') as f:
#     for i, user_id in enumerate(user_list):
#         f.write("{}::{}\n".format(i+1, user_id))


# generate item data

# data_dir01 = r"D:\Code\ML\texts\用户点击明细\组队商品浏览"
data_dir02 = r"D:\Code\ML\texts\用户点击明细\组队商品购买"


item_list = []
# for file_name in os.listdir(data_dir01):
#     t1 = time.time()
#
#     file_path = os.path.join(data_dir01, file_name)
#     dataframe = pd.read_excel(file_path)
#     item_list.append(list(set(dataframe['拼团编号'])))
#
#     t2 = time.time()
#     print(t2 - t1)
#
# print(item_list)

for file_name in os.listdir(data_dir02):
    t1 = time.time()

    file_path = os.path.join(data_dir02, file_name)
    dataframe = pd.read_excel(file_path)
    item_list.append(list(set(dataframe['拼团编号'])))

    t2 = time.time()
    print(t2 - t1)

print(item_list)
a = item_list

merged_list = [item for sublist in item_list for item in sublist]
item_list = sorted(list(set(merged_list)))

new_list = []
for i in item_list:
    if not np.isnan(i):
        new_list.append(int(i))

with open('item_buy.data', 'w') as f:
    for i, item_id in enumerate(new_list):
        f.write("{}::{}\n".format(i+1, item_id))


# generate rating data


# def get_data(rating, path):
#     dataframe = pd.read_excel(path)
#     print(len(dataframe))
#
#     # 删除nan数据然后重置索引
#     dataframe.dropna(subset=['拼团编号'], inplace=True)
#     dataframe.reset_index(drop=True, inplace=True)
#
#     merge_frame = dataframe['用户编号'].astype(str) + "::" + dataframe['拼团编号'].astype(int).astype(str)
#
#     rating = pd.concat([rating, merge_frame], axis=0)
#
#     return rating
#
#
# def merge_data01(rating, dir_path):
#
#     for file_name in os.listdir(dir_path):
#         t1 = time.time()
#
#         file_path = os.path.join(dir_path, file_name)
#         rating = get_data(rating, file_path)
#
#         t2 = time.time()
#         print(t2 - t1)
#
#     rating.reset_index(drop=True, inplace=True)
#     return rating
#
#
# # 浏览数据取loge的对数
# def merge_score(rating, my_log):
#     rating_set = set(rating)
#     print("集合数量: ", len(rating_set))
#
#     rating_array = rating.array
#     rating_data = []
#
#     for i, data in enumerate(rating_set):
#         num = len(rating_array[rating_array == data])
#         # 取对数后取小数点后三位
#         # rating_data.append(data + "::" + str(np.round(my_log(num), 3)))
#         rating_data.append(data + "::" + str(num))
#         print(i)
#     # 根据每个字符串分割后的第一个数值，进行排序
#     rating_data.sort(key=lambda x: int(x.split('::')[0]))
#
#     return rating_data
#
#
# if __name__ == '__main__':
#     dir_path01 = r"D:\Code\ML\texts\用户点击明细\组队商品浏览"
#     data_dir02 = r"D:\Code\ML\texts\用户点击明细\组队商品购买"
#
#     # rating = pd.Series()
#     # rating = merge_data01(rating, dir_path=dir_path01)
#     #
#     # rating_data01 = merge_score(rating, np.log)
#     # print(rating_data01)
#     # with open('rating_scan.data', 'w') as f:
#     #     for data in rating_data01:
#     #         f.write(data + '\n')
#
#     rating = pd.Series()
#     rating = merge_data01(rating, dir_path=data_dir02)
#
#     rating_data02 = merge_score(rating, np.log2)
#     with open('card_user_item/rating_buy.data', 'w') as f:
#         for data in rating_data02:
#             f.write(data + '\n')
#
