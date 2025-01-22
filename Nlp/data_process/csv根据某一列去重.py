import pandas as pd

df = pd.read_csv(r"D:\Code\ML\Text\test\paniniamerica_checklist_refresh.csv")

# 根据 subset 那一列进行去重
df.drop_duplicates(subset=['bgs_title'], keep='first', inplace=True)
df.reset_index(drop=True, inplace=True)
df.to_csv(r"D:\Code\ML\Text\test\paniniamerica_checklist_refresh_norepeat.csv", index=False)
print('end')