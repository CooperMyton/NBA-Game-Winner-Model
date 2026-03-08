import pandas as pd

existing = pd.read_csv("data/raw/TeamStatisticsAdvanced.csv")
new_df = pd.read_csv("data/raw/advanced_stats_new.csv")

# Map new column names to match existing CSV schema
col_map = {
    "TEAM_ID": "teamId",
    "TEAM_ABBREVIATION": "teamAbbreviation",
    "TEAM_NAME": "teamName",
    "GAME_ID": "gameId",
    "GAME_DATE": "gameDate",
    "MATCHUP": "matchup",
    "WL": "wl",
    "MIN": "min",
    "E_OFF_RATING": "eOffRating",
    "OFF_RATING": "offRating",
    "E_DEF_RATING": "eDefRating",
    "DEF_RATING": "defRating",
    "E_NET_RATING": "eNetRating",
    "NET_RATING": "netRating",
    "AST_PCT": "astPct",
    "AST_TO": "astTo",
    "AST_RATIO": "astRatio",
    "OREB_PCT": "orebPct",
    "DREB_PCT": "drebPct",
    "REB_PCT": "rebPct",
    "TM_TOV_PCT": "tmTovPct",
    "EFG_PCT": "efgPct",
    "TS_PCT": "tsPct",
    "E_PACE": "ePace",
    "PACE": "pace",
    "PACE_PER40": "pacePer40",
    "POSS": "poss",
    "PIE": "pie",
    "AVAILABLE_FLAG": "availableFlag",
    "SEASON_YEAR": "seasonYear"
}

new_df = new_df.rename(columns=col_map)

# Add missing columns that exist in original
new_df["gameDateTimeEst"] = pd.to_datetime(new_df["gameDate"])
new_df["home"] = new_df["matchup"].apply(lambda x: 1 if "vs." in str(x) else 0)
new_df["win"] = new_df["wl"].apply(lambda x: 1 if x == "W" else 0)
new_df["gameType"] = "Regular Season"

# Keep only columns that exist in existing CSV
keep_cols = [c for c in existing.columns if c in new_df.columns]
new_df = new_df[keep_cols]

# Remove duplicates (existing 2025/2026 data)
existing_ids = set(zip(existing["gameId"], existing["teamId"]))
new_df = new_df[~new_df.apply(lambda r: (r["gameId"], r["teamId"]) in existing_ids, axis=1)]

print(f"Adding {len(new_df)} new rows to existing {len(existing)}")

merged = pd.concat([existing, new_df], ignore_index=True)
merged["gameDateTimeEst"] = pd.to_datetime(merged["gameDateTimeEst"], errors="coerce")
merged = merged.sort_values("gameDateTimeEst")

merged.to_csv("data/raw/TeamStatisticsAdvanced_nn.csv", index=False)
print(f"Done. Total rows: {len(merged)}")