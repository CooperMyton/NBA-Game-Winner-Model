import pandas as pd
from src.feature_engineering import (
    load_and_merge,
    clean_columns,
    add_rolling_features,
    finalize_dataset
)

DEFAULT_START_DATE = pd.Timestamp("2023-10-01")
WINDOW = 13

def build_matchup_dataset(start_date=DEFAULT_START_DATE):
    df = load_and_merge(
        "data/raw/TeamStatistics.csv",
        "data/raw/TeamStatisticsAdvanced.csv",
        start_date
    )

    df = clean_columns(df)
    df = add_rolling_features(df, window=WINDOW)
    #df = add_rolling_features(df, window=5)
    df["days_rest"] = (
    df.groupby("teamId")["gameDateTimeEst"]
    .transform(lambda x: x.diff().dt.days.fillna(3))
)
    df = finalize_dataset(df)
    
    # Automatically detect rolling feature columns
    stats = [
    col for col in df.columns
    if "_last" in col and "netRating" not in col  # ADD "netRating" not in col
]



    df = df.sort_values("gameDateTimeEst")

    games = []

    for game_id, game in df.groupby("gameId"):

        if len(game) != 2:
            continue

        home_team = game[game["home"] == 1].iloc[0]
        away_team = game[game["home"] == 0].iloc[0]


        game_data = {}


        for stat in stats:
            game_data[f"{stat}_diff"] = home_team[stat] - away_team[stat]
            game_data["days_rest_home"] = home_team["days_rest"]
            game_data["days_rest_away"] = away_team["days_rest"]
            game_data["days_rest_diff"] = home_team["days_rest"] - away_team["days_rest"]


        game_data["home_win"] = int(home_team["win"])


        games.append(game_data)

    dataset = pd.DataFrame(games)

    dataset.to_csv("data/processed/matchup_dataset_v3.csv", index=False)
    
    print("Shape:", dataset.shape)
    print("Columns:", dataset.columns.tolist())
    print("Null counts:\n", dataset.isnull().sum())
    print("Class balance:\n", dataset["home_win"].value_counts())

    return dataset


if __name__ == "__main__":
    df = build_matchup_dataset()
    print(df.head())
