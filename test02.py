import pandas as pd

data_path = r"D:\Code\ML\Text\test\paniniamerica_checklist.csv"

data = pd.read_csv(data_path, encoding="utf-8")
data.reset_index(drop=True, inplace=True)


def get_merge_info(csv_id):
    year_data = f"{int(data['year'][csv_id])}-{int(str(data['year'][csv_id])[-2:]) + 1}"
    merge_data = f"{year_data} {data['program_new'][csv_id]} {data['card_set'][csv_id]} #{data['card_number'][csv_id]} {data['athlete'][csv_id]}"
    return merge_data


for i in range(10):
    data.loc[i, "bgs_title"] = get_merge_info(i)
    print(get_merge_info(i))