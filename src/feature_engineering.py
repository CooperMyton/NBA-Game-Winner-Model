#compute rolling averages, opponent merge, diff features, feature selection

import pandas as pd

# Columns to remove
leakage_cols = [
    "win_adv", "wl",
    "teamScore", "opponentScore", "plusMinusPoints",
    "q1Points", "q2Points", "q3Points", "q4Points",
    "benchPoints", "biggestLead", "biggestScoringRun",
    "leadChanges", "timesTied",
    "pointsFastBreak", "pointsFromTurnovers",
    "pointsInThePaint", "pointsSecondChance",
    "timeoutsRemaining",
    "numMinutes", "min", "poss",
    "gameDate", "gameDateTimeEst_adv",
    "matchup"
]

junk_cols = [
    "coachId",
    "teamCity_adv", "teamName_adv",
    "opponentTeamCity_adv", "opponentTeamName_adv",
    "teamName_right",
    "home_adv"
]

#taking gameId out for tree model
id_cols = [
    "teamCity", "teamName",
    "opponentTeamCity", "opponentTeamName",
    "opponentTeamId",
    "teamAbbreviation"
]

raw_game_stats = [
    "assists", "blocks", "steals",
    "fieldGoalsAttempted", "fieldGoalsMade", "fieldGoalsPercentage",
    "threePointersAttempted", "threePointersMade", "threePointersPercentage",
    "freeThrowsAttempted", "freeThrowsMade", "freeThrowsPercentage",
    "reboundsDefensive", "reboundsOffensive", "reboundsTotal",
    "foulsPersonal", "turnovers",
    "astPct", "astRatio", "astTo",
    "defRating", "drebPct", "eDefRating", "eNetRating", "eOffRating",
    "ePace", "efgPct", "netRating", "offRating",
    "orebPct", "pace", "pacePer40", "pie", "rebPct", "tmTovPct", "tsPct"
]


#Load csv and merge the two csvs together
def load_and_merge(games_path, advanced_path, start_date):
    games = pd.read_csv(games_path)
    advanced = pd.read_csv(advanced_path)

    games["gameDateTimeEst"] = pd.to_datetime(games["gameDateTimeEst"])
    advanced["gameDateTimeEst"] = pd.to_datetime(advanced["gameDateTimeEst"], errors="coerce")

    games = games.sort_values(["teamId", "gameDateTimeEst"])
    advanced = advanced.sort_values(["teamId", "gameDateTimeEst"])

    games = games[games["teamId"] >= 1610612700]
    advanced = advanced[advanced["teamId"] >= 1610612700]

    games = games[games["gameDateTimeEst"] >= start_date]
    advanced = advanced[advanced["gameDateTimeEst"] >= start_date]

    df = games.merge(
        advanced,
        on=["gameId", "teamId"],
        how="left",
        suffixes=("", "_adv")
    )

    return df

#clean columns
def clean_columns(df):
    df = df.drop(columns=leakage_cols + junk_cols + id_cols)
    df = df[df["gameType"] == "Regular Season"]
    df = df.drop(columns=["gameType"])
    return df

#rolling features for n games
def add_rolling_features(df, window=5):
    df = df.sort_values("gameDateTimeEst")

    rolling_stats = [
        "assists", "reboundsTotal", "turnovers",
        "fieldGoalsPercentage", "threePointersPercentage",
        "freeThrowsPercentage",
        "offRating", "defRating", "netRating",
        "pace", "tsPct"
    ]

    for stat in rolling_stats:
        df[f"{stat}_last{window}"] = (
            df.groupby("teamId")[stat]
              .transform(lambda x: x.shift(1).rolling(window).mean())
        )

    return df

#opponent diff features
def add_opponent_diffs(df, window=5):
    rolling_cols = [col for col in df.columns if f"last{window}" in col]

    opponent_df = df[["gameDateTimeEst", "teamId"] + rolling_cols].copy()
    opponent_df = opponent_df.rename(columns={"teamId": "opponentTeamId_adv"})

    for col in rolling_cols:
        opponent_df = opponent_df.rename(columns={col: col + "_opp"})

    df = df.merge(
        opponent_df,
        on=["gameDateTimeEst", "opponentTeamId_adv"],
        how="left"
    )

    for col in rolling_cols:
        df[col + "_diff"] = df[col] - df[col + "_opp"]

    cols_to_drop = rolling_cols + [c + "_opp" for c in rolling_cols]
    df = df.drop(columns=cols_to_drop)

    return df


#final cleanup
def finalize_dataset(df):
    df = df.drop(columns=raw_game_stats)
    df = df.dropna()
    return df
