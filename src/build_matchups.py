#compute rolling stats per team, merge home and away into single row, produce clean training
import pandas as pd
from src.feature_engineering import (
    load_and_merge,
    clean_columns,
    add_rolling_features,
    finalize_dataset
)

START_DATE = pd.Timestamp("2023-10-01")


def build_matchup_dataset(window=5):

    df = load_and_merge(
        "data/raw/TeamStatistics.csv",
        "data/raw/TeamStatisticsAdvanced.csv",
        START_DATE
    )

    df = clean_columns(df)
    df = add_rolling_features(df, window=window)
    df = finalize_dataset(df)

    # Keep only rolling features + identifiers
    rolling_cols = [c for c in df.columns if f"_last{window}" in c]

    keep_cols = ["gameId", "teamId", "home", "win", "gameDateTimeEst"] + rolling_cols
    df = df[keep_cols]

    # Separate home and away
    home_df = df[df["home"] == 1].copy()
    away_df = df[df["home"] == 0].copy()

    # Rename columns
    home_df = home_df.rename(columns=lambda x: f"home_{x}" if x in rolling_cols else x)
    away_df = away_df.rename(columns=lambda x: f"away_{x}" if x in rolling_cols else x)

    # Merge on gameId
    matchups = home_df.merge(
        away_df,
        left_on="gameId",
        right_on="gameId"
    )

    # Target: did home team win?
    matchups["home_win"] = matchups["win_x"]

    return matchups
