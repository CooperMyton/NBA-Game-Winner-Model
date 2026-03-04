# load saved model, take two teams, generate features, return probability
import pandas as pd
import joblib
from src.feature_engineering import (
    load_and_merge,
    clean_columns,
    add_rolling_features,
    add_opponent_diffs,
    finalize_dataset
)

START_DATE = pd.Timestamp("2023-10-01")

model = joblib.load("models/logistic_model.pkl")

#print("Model expects", len(model.feature_names_in_), "features")
#print(model.feature_names_in_)
#exit()




VALID_TEAMS = [
    "hawks", "celtics", "nets", "hornets", "bulls",
    "cavaliers", "mavericks", "nuggets", "pistons",
    "warriors", "rockets", "pacers", "clippers",
    "lakers", "grizzlies", "heat", "bucks",
    "timberwolves", "pelicans", "knicks", "thunder",
    "magic", "76ers", "suns", "blazers",
    "kings", "spurs", "raptors", "jazz", "wizards"
]


# given 2 teams and a rolling average window, predict a winner
def predict_matchup(team1, team2, window=5):

    team1 = team1.lower()
    team2 = team2.lower()

    # -----------------------------
    # Validation
    # -----------------------------
    if team1 not in VALID_TEAMS:
        raise ValueError(f"{team1} is not a valid NBA team name.")

    if team2 not in VALID_TEAMS:
        raise ValueError(f"{team2} is not a valid NBA team name.")

    if team1 == team2:
        raise ValueError("Teams must be different.")

    # -----------------------------
    # Load dataset (raw names still exist here)
    # -----------------------------
    df = load_and_merge(
        "data/raw/TeamStatistics.csv",
        "data/raw/TeamStatisticsAdvanced.csv",
        START_DATE
    )

    # Build team mapping BEFORE cleaning
    team_mapping = (
        df[["teamId", "teamName"]]
        .drop_duplicates()
    )
    team_mapping["teamName_lower"] = team_mapping["teamName"].str.lower()

    team1_id = team_mapping.loc[
        team_mapping["teamName_lower"] == team1, "teamId"
    ].values[0]

    team2_id = team_mapping.loc[
        team_mapping["teamName_lower"] == team2, "teamId"
    ].values[0]

    # -----------------------------
    # Rebuild full feature pipeline
    # -----------------------------
    df = clean_columns(df)
    df = add_rolling_features(df, window=window)
    df = add_opponent_diffs(df, window=window)
    df = finalize_dataset(df)

    # -----------------------------
    # Get most recent row per team (USE teamId now)
    # -----------------------------
    latest = (
        df.sort_values("gameDateTimeEst")
        .groupby("teamId")
        .tail(1)
    )

    team1_row = latest[latest["teamId"] == team1_id]
    team2_row = latest[latest["teamId"] == team2_id]

    if team1_row.empty or team2_row.empty:
        raise ValueError("One of the teams has insufficient data.")

    # -----------------------------
    # Build feature vector
    # -----------------------------
    feature_cols = model.feature_names_in_


    features = team1_row[feature_cols]


    # -----------------------------
    # Predict probability
    # -----------------------------
    prob = model.predict_proba(features)[0][1]

    if prob > 0.5:
        winner = team1
        confidence = prob
    else:
        winner = team2
        confidence = 1 - prob

    return winner.title(), round(confidence, 3)


if __name__ == "__main__":
    t1 = input("Enter first team: ")
    t2 = input("Enter second team: ")

    winner, confidence = predict_matchup(t1, t2)

    print(f"\nPredicted Winner: {winner}")
    print(f"Confidence: {confidence}")
