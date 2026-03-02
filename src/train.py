#split, train, evaluate, save model
from src.feature_engineering import (
    load_and_merge,
    clean_columns,
    add_rolling_features,
    add_opponent_diffs,
    finalize_dataset
)

from src.models import get_logistic

import pandas as pd

start_date = pd.Timestamp("2023-10-01")

df = load_and_merge("data/raw/TeamStatistics.csv", "data/raw/TeamStatisticsAdvanced.csv", start_date)
df = clean_columns(df)
df = add_rolling_features(df, window=5)
df = add_opponent_diffs(df, window=5)
df = finalize_dataset(df)

df = df.sort_values("gameDateTimeEst")

split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train = train_df.drop(columns=["win", "gameDateTimeEst", "teamId"])
y_train = train_df["win"]

X_test = test_df.drop(columns=["win", "gameDateTimeEst", "teamId"])
y_test = test_df["win"]

model = get_logistic()
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))
