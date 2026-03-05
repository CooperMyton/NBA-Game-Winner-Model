import pandas as pd
import joblib

from src.feature_engineering import (
    load_and_merge,
    clean_columns,
    add_rolling_features,
    finalize_dataset
)

START_DATE = pd.Timestamp("2023-10-01")

model = joblib.load("models/xgb_model.pkl")

VALID_TEAMS = [
    "hawks", "celtics", "nets", "hornets", "bulls",
    "cavaliers", "mavericks", "nuggets", "pistons",
    "warriors", "rockets", "pacers", "clippers",
    "lakers", "grizzlies", "heat", "bucks",
    "timberwolves", "pelicans", "knicks", "thunder",
    "magic", "76ers", "suns", "blazers",
    "kings", "spurs", "raptors", "jazz", "wizards"
]

TEAM_NAME_TO_ID = {
    "hawks": 1610612737,
    "celtics": 1610612738,
    "nets": 1610612751,
    "hornets": 1610612766,
    "bulls": 1610612741,
    "cavaliers": 1610612739,
    "mavericks": 1610612742,
    "nuggets": 1610612743,
    "pistons": 1610612765,
    "warriors": 1610612744,
    "rockets": 1610612745,
    "pacers": 1610612754,
    "clippers": 1610612746,
    "lakers": 1610612747,
    "grizzlies": 1610612763,
    "heat": 1610612748,
    "bucks": 1610612749,
    "timberwolves": 1610612750,
    "pelicans": 1610612740,
    "knicks": 1610612752,
    "thunder": 1610612760,
    "magic": 1610612753,
    "76ers": 1610612755,
    "suns": 1610612756,
    "blazers": 1610612757,
    "kings": 1610612758,
    "spurs": 1610612759,
    "raptors": 1610612761,
    "jazz": 1610612762,
    "wizards": 1610612764
}


def predict_matchup(home_team, away_team, window=13):

    home_team = home_team.lower()
    away_team = away_team.lower()

    if home_team not in VALID_TEAMS:
        raise ValueError(f"{home_team} is not valid.")

    if away_team not in VALID_TEAMS:
        raise ValueError(f"{away_team} is not valid.")

    if home_team == away_team:
        raise ValueError("Teams must be different.")

    # Load and rebuild rolling stats
    df = load_and_merge(
        "data/raw/TeamStatistics.csv",
        "data/raw/TeamStatisticsAdvanced.csv",
        START_DATE
    )

    df = clean_columns(df)
    df = add_rolling_features(df, window=window)
    df = finalize_dataset(df)

    # Get most recent game per team
    latest = (
        df.sort_values("gameDateTimeEst")
          .groupby("teamId")
          .tail(1)
    )

    # Map team names to teamId
    home_id = TEAM_NAME_TO_ID[home_team]
    away_id = TEAM_NAME_TO_ID[away_team]

    home_row = latest[latest["teamId"] == home_id]
    away_row = latest[latest["teamId"] == away_id]

    if home_row.empty or away_row.empty:
        raise ValueError("Insufficient data for one team.")

    # Build feature vector
    feature_cols = model.feature_names_in_

    feature_dict = {}

    for col in feature_cols:
        if col.startswith("home_"):
            stat = col.replace("home_", "")
            feature_dict[col] = home_row[stat].values[0]

        elif col.startswith("away_"):
            stat = col.replace("away_", "")
            feature_dict[col] = away_row[stat].values[0]

    features = pd.DataFrame([feature_dict])[feature_cols]

    prob = model.predict_proba(features)[0][1]

    if prob >= 0.5:
        winner = home_team
        confidence = prob
    else:
        winner = away_team
        confidence = 1 - prob

    return winner.title(), round(confidence, 3)


if __name__ == "__main__":
    home = input("Enter home team: ")
    away = input("Enter away team: ")

    winner, confidence = predict_matchup(home, away)

    print(f"\nPredicted Winner: {winner}")
    print(f"Confidence: {confidence}")
