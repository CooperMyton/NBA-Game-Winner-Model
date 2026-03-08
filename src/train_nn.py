import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from src.build_matchups_v3 import build_matchup_dataset

import joblib
import os

df = build_matchup_dataset(start_date=pd.Timestamp("2020-10-01"))

X = df.drop(columns=["home_win", "game_id"])
y = df["home_win"]

#from sklearn.ensemble import RandomForestClassifier
#rf = RandomForestClassifier(n_estimators=100, random_state=42)
#rf.fit(X, y)

#importances = pd.Series(rf.feature_importances_, index=X.columns)
#print(importances.sort_values(ascending=False))
#exit()


#print("Shape:", df.shape)
#print(df.head())
#print(df.drop(columns=["home_win"]).columns.tolist())
#exit()
#print("Shape:", df.shape)
#print("Columns:", df.drop(columns=["home_win"]).columns.tolist())
#print("Date range:", df["gameDateTimeEst"].min(), "to", df["gameDateTimeEst"].max())
#print("Null counts:\n", df.isnull().sum())
#print("Class balance:\n", df["home_win"].value_counts())
#exit()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/nn_scaler.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    shuffle=False
)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=15,
    restore_best_weights=True
)

# Handle class imbalance
neg, pos = (y == 0).sum(), (y == 1).sum()
class_weight = {0: 1.0, 1: neg / pos} # down-weight the majority class

from collections import Counter
print("Train class balance:", Counter(y_train))
print("Test class balance:", Counter(y_test))

model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    class_weight=class_weight
)

preds = (model.predict(X_test) > 0.5).astype(int)

accuracy = accuracy_score(y_test, preds)

print("Test Accuracy:", accuracy)

model.save("models/nn_model.keras")
