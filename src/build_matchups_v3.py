import pandas as pd
from src.feature_engineering import (
    load_and_merge,
    clean_columns,
    add_rolling_features,
    finalize_dataset
)

DEFAULT_START_DATE = pd.Timestamp("2023-10-01")
WINDOW = 13

def compute_elo(games_path, k=25, home_advantage=85):
    """Compute pre-game ELO ratings for every team-game from all history."""
    df = pd.read_csv(games_path)
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"])
    df = df[df["teamId"] >= 1610612700]
    df = df.sort_values("gameDateTimeEst")

    elo = {}  # teamId -> current ELO
    records = []  # (gameId, teamId, pre-game ELO)

    for game_id, game in df.groupby("gameId"):
        if len(game) != 2:
            continue

        home = game[game["home"] == 1].iloc[0]
        away = game[game["home"] == 0].iloc[0]

        home_id = home["teamId"]
        away_id = away["teamId"]

        # Initialize ELO if first appearance
        home_elo = elo.get(home_id, 1500)
        away_elo = elo.get(away_id, 1500)

        # Store pre-game ELO
        records.append({"gameId": game_id, "teamId": home_id, "elo": home_elo})
        records.append({"gameId": game_id, "teamId": away_id, "elo": away_elo})

        # Expected win probabilities
        exp_home = 1 / (1 + 10 ** ((away_elo - (home_elo + home_advantage)) / 400))
        exp_away = 1 - exp_home

        # Actual outcomes
        actual_home = int(home["win"])
        actual_away = 1 - actual_home

        # Update ELO
        elo[home_id] = home_elo + k * (actual_home - exp_home)
        elo[away_id] = away_elo + k * (actual_away - exp_away)

    return pd.DataFrame(records)

def build_matchup_dataset(start_date=DEFAULT_START_DATE):
    df = load_and_merge(
        "data/raw/TeamStatistics.csv",
        "data/raw/TeamStatisticsAdvanced_nn.csv",
        start_date
    )

    df = clean_columns(df)
    df = add_rolling_features(df, window=WINDOW)
    
    df["season_win_pct"] = (
        df.groupby("teamId")["win"]
        .transform(lambda x: x.shift(1).expanding().mean())
)
    
    cols_to_drop = ["opponentTeamId_adv", "seasonWins", "seasonLosses"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    df = df.ffill()
    df = add_rolling_features(df, window=5)
    df = finalize_dataset(df)
    
    # Compute ELO from ALL history and merge in
    elo_df = compute_elo("data/raw/TeamStatistics.csv")
    df = df.merge(elo_df, on=["gameId", "teamId"], how="left")
    
    # Automatically detect rolling feature columns
    stats = [
    col for col in df.columns
    if "_last" in col 
    and "netRating" not in col
    and col != "win_last5"  # low importance
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
            

        game_data["season_win_pct_diff"] = home_team["season_win_pct"] - away_team["season_win_pct"]
        game_data["elo_diff"] = home_team["elo"] - away_team["elo"]
        game_data["game_id"] = game_id 
        game_data["home_win"] = int(home_team["win"])
        


        games.append(game_data)

    dataset = pd.DataFrame(games)
    dataset.to_csv("data/processed/matchup_dataset_v3.csv", index=False)
    
    #print("Shape:", dataset.shape)
    #print("Columns:", dataset.columns.tolist())
    #print("Null counts:\n", dataset.isnull().sum())
    #print("Class balance:\n", dataset["home_win"].value_counts())

    return dataset


if __name__ == "__main__":
    df = build_matchup_dataset()
    print(df.head())
