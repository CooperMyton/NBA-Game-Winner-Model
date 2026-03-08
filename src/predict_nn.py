import pandas as pd
import tensorflow as tf
import joblib
from src.build_matchups_v3 import build_matchup_dataset

scaler = joblib.load("models/nn_scaler.pkl")
model = tf.keras.models.load_model("models/nn_model.keras")

df = pd.read_csv("data/processed/matchup_dataset_v3.csv")
feature_cols = [c for c in df.columns if c not in ["home_win", "game_id"]]

TEAM_NAME_TO_ID = {
    "hawks": 1610612737, "celtics": 1610612738, "nets": 1610612751,
    "hornets": 1610612766, "bulls": 1610612741, "cavaliers": 1610612739,
    "mavericks": 1610612742, "nuggets": 1610612743, "pistons": 1610612765,
    "warriors": 1610612744, "rockets": 1610612745, "pacers": 1610612754,
    "clippers": 1610612746, "lakers": 1610612747, "grizzlies": 1610612763,
    "heat": 1610612748, "bucks": 1610612749, "timberwolves": 1610612750,
    "pelicans": 1610612740, "knicks": 1610612752, "thunder": 1610612760,
    "magic": 1610612753, "76ers": 1610612755, "suns": 1610612756,
    "blazers": 1610612757, "kings": 1610612758, "spurs": 1610612759,
    "raptors": 1610612761, "jazz": 1610612762, "wizards": 1610612764
}


def predict_matchup(home_team, away_team):
    home_team = home_team.lower()
    away_team = away_team.lower()

    if home_team not in TEAM_NAME_TO_ID:
        raise ValueError(f"{home_team} is not a valid team.")
    if away_team not in TEAM_NAME_TO_ID:
        raise ValueError(f"{away_team} is not a valid team.")
    if home_team == away_team:
        raise ValueError("Teams must be different.")

    # Rebuild full dataset to get latest rolling stats and ELO
    full_df = build_matchup_dataset(start_date=pd.Timestamp("2020-10-01"))
    full_df = full_df.sort_values("game_id")

    # Get most recent game for each team
    # We need to find the last game each team appeared in
    raw_df = pd.read_csv("data/processed/matchup_dataset_v3.csv")

    home_id = TEAM_NAME_TO_ID[home_team]
    away_id = TEAM_NAME_TO_ID[away_team]

    # Load the team-level data to get latest stats per team
    from src.feature_engineering import load_and_merge, clean_columns, add_rolling_features, finalize_dataset
    from src.build_matchups_v3 import compute_elo

    START_DATE = pd.Timestamp("2020-10-01")
    team_df = load_and_merge(
        "data/raw/TeamStatistics.csv",
        "data/raw/TeamStatisticsAdvanced_nn.csv",
        START_DATE
    )
    team_df = clean_columns(team_df)
    team_df = add_rolling_features(team_df, window=13)
    team_df["season_win_pct"] = (
        team_df.groupby("teamId")["win"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    cols_to_drop = ["opponentTeamId_adv", "seasonWins", "seasonLosses"]
    team_df = team_df.drop(columns=[c for c in cols_to_drop if c in team_df.columns])
    team_df = team_df.ffill()
    team_df = add_rolling_features(team_df, window=5)
    team_df = finalize_dataset(team_df)

    elo_df = compute_elo("data/raw/TeamStatistics.csv")
    team_df = team_df.merge(elo_df, on=["gameId", "teamId"], how="left")

    latest = team_df.sort_values("gameDateTimeEst").groupby("teamId").tail(1)

    home_row = latest[latest["teamId"] == home_id]
    away_row = latest[latest["teamId"] == away_id]

    if home_row.empty or away_row.empty:
        raise ValueError("Insufficient data for one or both teams.")

    # Build feature vector matching training features
    stat_cols = [c for c in feature_cols if "_diff" not in c and c != "season_win_pct_diff" and c != "elo_diff"]
    
    feature_dict = {}
    for col in feature_cols:
        if col == "elo_diff":
            feature_dict[col] = home_row["elo"].values[0] - away_row["elo"].values[0]
        elif col == "season_win_pct_diff":
            feature_dict[col] = home_row["season_win_pct"].values[0] - away_row["season_win_pct"].values[0]
        elif col.endswith("_diff"):
            stat = col.replace("_diff", "")
            if stat in home_row.columns:
                feature_dict[col] = home_row[stat].values[0] - away_row[stat].values[0]
            else:
                feature_dict[col] = 0

    X = pd.DataFrame([feature_dict])[feature_cols]
    X_scaled = scaler.transform(X)
    prob = model.predict(X_scaled)[0][0]

    if prob >= 0.5:
        return home_team.title(), round(float(prob), 3)
    else:
        return away_team.title(), round(float(1 - prob), 3)


if __name__ == "__main__":
    home = input("Enter home team: ")
    away = input("Enter away team: ")
    winner, confidence = predict_matchup(home, away)
    print(f"\nPredicted Winner: {winner}")
    print(f"Confidence: {confidence}")
