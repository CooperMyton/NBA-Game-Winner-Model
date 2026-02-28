#trying to get kaggle dataset to interact
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#List of attributes which are unnecessary
COLS_TO_DELETE = ["gameId", "teamID", "opponentTeamId", "coachId", "timeoutsRemaining", "timesTied"]

start_date = pd.Timestamp("2023-10-01")

games = pd.read_csv('TeamStatistics.csv')
advanced = pd.read_csv('TeamStatisticsAdvanced.csv')

#Converting Dates and sorting
games["gameDateTimeEst"] = pd.to_datetime(games["gameDateTimeEst"])
advanced["gameDateTimeEst"] = pd.to_datetime(advanced['gameDateTimeEst'], errors="coerce")

#Sort by team id to get rid of weird teams
games = games.sort_values(["teamId", "gameDateTimeEst"])
advanced = advanced.sort_values(["teamId", "gameDateTimeEst"])
games = games[games["teamId"] >= 1610612700]
advanced = advanced[advanced["teamId"] >= 1610612700]

#Get rid of games before start date
games = games[games["gameDateTimeEst"] >= start_date]
advanced = advanced[advanced["gameDateTimeEst"] >= start_date]

#left merge games w advanced
df = games.merge(
    advanced,
    on=["gameId", "teamId"],
    how="left",
    suffixes=("", "_adv")
)

#define target, bad and good features
target = "win"
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
id_cols = [
    "gameId",
    "teamCity", "teamName",
    "opponentTeamCity", "opponentTeamName",
    "opponentTeamId",
    "teamAbbreviation"
]
#drop bad ones
df = df.drop(columns=leakage_cols + junk_cols + id_cols)

#Drop preseason
df = df[df["gameType"] == "Regular Season"]
df = df.drop(columns=["gameType"])

#Create rolling stats for analysis
rolling_stats = [
    "assists", "reboundsTotal", "turnovers",
    "fieldGoalsPercentage", "threePointersPercentage",
    "freeThrowsPercentage",
    "offRating", "defRating", "netRating",
    "pace", "tsPct"
]
df = df.sort_values("gameDateTimeEst")
for stat in rolling_stats:
    df[f"{stat}_last5"] = (
        df.groupby("teamId")[stat].transform(lambda x: x.shift(1).rolling(5).mean())
    )
    
#remove pergame stats
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
df = df.drop(columns= raw_game_stats)
df = df.dropna() #get rid of any row with NA attribute

#Time based training
df = df.sort_values("gameDateTimeEst")
split_idx = int(len(df) * 0.8)

train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train = train_df.drop(columns = ["win", "gameDateTimeEst", "teamId"])
y_train = train_df["win"]

X_test = test_df.drop(columns = ["win", "gameDateTimeEst", "teamId"])
y_test = test_df["win"]


#First model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(F"Test accuracy: {acc:.3f}")

#Sanity checks
#print(df[[f"{s}_last5" for s in rolling_stats]].head(10))
#print(df.isna().sum().sum())
#print(len(df))
#print(df.columns)
#print(games.duplicated(subset={"gameId", "teamId"}).sum())
#print(advanced.duplicated(subset=["gameId", "teamId"]).sum())
#print(games["gameDateTimeEst"].min(), games["gameDateTimeEst"].max())
#print(len(games),len(advanced))
#print(games["teamId"].min())
#print(advanced["teamId"].min())
#print(games["gameDateTimeEst"].min(), games["gameDateTimeEst"].max())
#print(advanced["gameDateTimeEst"].min(), advanced["gameDateTimeEst"].max())
#print(games[["teamId", "gameDateTimeEst"]].head(10))
#print(games.head())
#print(advanced.head())
#print(games.info())
#print(advanced.info())