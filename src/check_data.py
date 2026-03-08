import pandas as pd
adv = pd.read_csv("data/raw/TeamStatisticsAdvanced_nn.csv")
adv["gameDateTimeEst"] = pd.to_datetime(adv["gameDateTimeEst"], errors="coerce")
adv["year"] = adv["gameDateTimeEst"].dt.year
print(adv.groupby("year").size())