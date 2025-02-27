from thefuzz import fuzz
from thefuzz import process
import pandas as pd
import time

data_path = r"D:\Code\ML\Project\Card_Text_Parse\Data\program_cardSet_athlete.csv"
data = pd.read_csv(data_path)

print()

# database = list(data['program'].dropna().unique())
database = list(data['card_set'].dropna().unique())
# database = list(data['athlete'].dropna().unique())

query = "Jalen Hood-Schifino 2023-24 Panini Mosaic Silver Mosaic RC Los Angeles Lakers"  # 包含错误和顺序变化

t1 = time.time()
# 使用 process.extract 找到多个匹配
matches = process.extract(query, database, scorer=fuzz.partial_ratio, limit=20)
print(f"Top 5 matches: {matches}")
print('--- %s seconds ---' % (time.time() - t1))

