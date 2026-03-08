import pandas as pd
df = pd.read_csv("data/raw/TeamStatisticsAdvanced.csv")
print(df.columns.tolist())
print(df.head(2))