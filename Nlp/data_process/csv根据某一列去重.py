import pandas as pd

df = pd.read_csv(r"C:\Code\ML\Text\psa2.csv")

# 根据 subset 那一列进行去重
df.drop_duplicates(subset=['tital'], keep='first', inplace=True)
df.reset_index(drop=True, inplace=True)
df.to_csv(r"C:\Code\ML\Text\psa2_distill.csv", index=False)
print('end')