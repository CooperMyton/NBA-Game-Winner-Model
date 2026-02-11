#trying to get kaggle dataset to interact
import pandas as pd
import numpy as np

df = pd.read_csv('TeamStatistics.csv')

df = df.drop("gameId", axis=1, inplace=True)
