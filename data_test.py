#trying to get kaggle dataset to interact
import pandas as pd
import numpy as np

#List of attributes which are unnecessary
COLS_TO_DELETE = ["gameId", "teamID", "opponentTeamId", "coachId", "timeoutsRemaining", "timesTied"]

games = pd.read_csv('TeamStatistics.csv')
advanced = pd.read_csv('TeamStatisticsAdvanced.csv')

#Converting Dates and sorting
games["gameDateTimeEst"] = pd.to_datetime(games["gameDateTimeEst"])
advanced["gameDateTimeEst"] = pd.to_datetime(advanced['gameDateTimeEst'], errors="coerce")

games = games.sort_values(["teamId", "gameDateTimeEst"])
advanced = advanced.sort_values(["teamId", "gameDateTimeEst"])


#Sanity checks
print(games[["teamId", "gameDateTimeEst"]].head(10))
#print(games.head())
#print(advanced.head())
#print(games.info())
#print(advanced.info())