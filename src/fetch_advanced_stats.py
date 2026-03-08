import pandas as pd
import time
from nba_api.stats.endpoints import teamgamelogs
from nba_api.stats.static import teams

SEASONS = [
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"
]

all_teams = teams.get_teams()
team_ids = [t["id"] for t in all_teams]

existing = pd.read_csv("data/raw/TeamStatisticsAdvanced.csv")
print(f"Existing rows: {len(existing)}")

new_rows = []

for season in SEASONS:
    print(f"\nFetching {season}...")
    for team_id in team_ids:
        try:
            logs = teamgamelogs.TeamGameLogs(
                team_id_nullable=team_id,
                season_nullable=season,
                measure_type_player_game_logs_nullable="Advanced"
            )
            df = logs.get_data_frames()[0]
            new_rows.append(df)
            time.sleep(0.6)  # avoid rate limiting
        except Exception as e:
            print(f"  Failed {team_id} {season}: {e}")
            time.sleep(2)

new_df = pd.concat(new_rows, ignore_index=True)
print(f"\nFetched {len(new_df)} new rows")
print("New columns:", new_df.columns.tolist())

# Save raw pull separately first so you can inspect it
new_df.to_csv("data/raw/advanced_stats_new.csv", index=False)
print("Saved to data/raw/advanced_stats_new.csv")
