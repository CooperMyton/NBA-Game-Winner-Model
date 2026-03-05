#train for tree model
from src.build_matchups import build_matchup_dataset
from xgboost import XGBClassifier
import pandas as pd
import joblib
import os

df = build_matchup_dataset(window=5)
#print(df.columns)
#exit()

df = df.sort_values("gameDateTimeEst_x")

split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

drop_cols = [
    "gameId",
    "gameDateTimeEst_x",
    "teamId_x",
    "teamId_y",
    "home_x",
    "home_y",
    "win_x",
    "win_y",
    "home_win",
    "gameDateTimeEst_y"
]

X_train = train_df.drop(columns=drop_cols)
y_train = train_df["home_win"]

X_test = test_df.drop(columns=drop_cols)
y_test = test_df["home_win"]

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)

print("New Model Accuracy:", model.score(X_test, y_test))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_model.pkl")
