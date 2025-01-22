import pandas as pd

data_path = r"D:\Code\ML\Text\test\paniniamerica_checklist.csv"

data = pd.read_csv(data_path, encoding="utf-8")


# def get_merge_info(csv_id):
#     year_data = f"{int(data['year'][csv_id])}-{int(str(data['year'][csv_id])[-2:]) + 1}"
#     merge_data = f"{year_data} {data['program_new'][csv_id]} {data['card_set'][csv_id]} #{data['card_number'][csv_id]} {data['athlete'][csv_id]}"
#     return merge_data
#
#
# # 使用 apply 函数
# data["bgs_title"] = data.apply(get_merge_info, axis=1)
# print(data["bgs_title"])

# ===================================
# 向量化操作
year_data = data['year'].astype(str) + '-' + (data['year'] % 100 + 1).astype(str).str.zfill(2)
data["bgs_title"] = year_data + ' ' + data['program_new'] + ' ' + data['card_set'] + ' #' + data['card_number'].astype(str) + ' ' + data['athlete']

print(data["bgs_title"])


data.to_csv(r"D:\Code\ML\Text\test\paniniamerica_checklist_refresh.csv", index=False)