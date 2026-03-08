import pandas as pd
import tensorflow as tf
import joblib

from src.build_matchups_v3 import build_matchup_dataset

scaler = joblib.load("models/nn_scaler.pkl")

model = tf.keras.models.load_model("models/nn_model.keras")

df = pd.read_csv("data/processed/matchup_dataset_v3.csv")

feature_cols = df.drop(columns=["home_win"]).columns


def predict(features):

    X = pd.DataFrame([features])[feature_cols]
    X_scaled = scaler.transform(X)
    prob = model.predict(X_scaled)[0][0]

    if prob >= 0.5:
        return "Home Team", prob
    else:
        return "Away Team", 1 - prob
