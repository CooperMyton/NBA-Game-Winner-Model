#trying to get kaggle dataset to interact
import pandas as pd
import numpy as np

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


#Sanity checks
print(len(df))
print(df.columns)
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