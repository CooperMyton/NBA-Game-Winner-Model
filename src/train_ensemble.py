import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.build_matchups_v3 import build_matchup_dataset
from src.build_matchups import build_matchup_dataset as build_matchup_dataset_v2

# Load NN dataset and models
nn_df = build_matchup_dataset(start_date=pd.Timestamp("2020-10-01"))
nn_scaler = joblib.load("models/nn_scaler.pkl")
nn_model = tf.keras.models.load_model("models/nn_model.keras")

# Load XGBoost dataset and model
xgb_df = build_matchup_dataset_v2(window=13)
xgb_model = joblib.load("models/xgb_model.pkl")

# Align both datasets on gameId
xgb_df = xgb_df.sort_values("gameDateTimeEst_x")
nn_df_sorted = nn_df.copy()

# Get NN probabilities
nn_X = nn_df_sorted.drop(columns=["home_win"])
nn_X_scaled = nn_scaler.transform(nn_X)
nn_probs = nn_model.predict(nn_X_scaled).flatten()

# Get XGBoost probabilities - need matching games
xgb_feature_cols = xgb_model.feature_names_in_
drop_cols = [
    "gameId", "gameDateTimeEst_x", "teamId_x", "teamId_y",
    "home_x", "home_y", "win_x", "win_y", "home_win", "gameDateTimeEst_y"
]
xgb_X = xgb_df.drop(columns=drop_cols)
xgb_probs = xgb_model.predict_proba(xgb_X)[:, 1]
xgb_y = xgb_df["home_win"]

# Use only the overlapping date range
# NN has more history so we align to XGBoost's length
nn_probs_aligned = nn_probs[-len(xgb_probs):]
nn_y = nn_df_sorted["home_win"].values[-len(xgb_probs):]

# Build meta-features
meta_X = np.column_stack([nn_probs_aligned, xgb_probs])
meta_y = xgb_y.values

# Time-based split
split_idx = int(len(meta_X) * 0.8)
X_train, X_test = meta_X[:split_idx], meta_X[split_idx:]
y_train, y_test = meta_y[:split_idx], meta_y[split_idx:]

# Train meta-model
meta_model = LogisticRegression()
meta_model.fit(X_train, y_train)

preds = meta_model.predict(X_test)
print("Ensemble Test Accuracy:", accuracy_score(y_test, preds))
print("NN alone:", accuracy_score(y_test, (nn_probs_aligned[split_idx:] >= 0.5).astype(int)))
print("XGBoost alone:", accuracy_score(y_test, (xgb_probs[split_idx:] >= 0.5).astype(int)))

joblib.dump(meta_model, "models/ensemble_model.pkl")
