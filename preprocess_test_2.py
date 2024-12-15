import pandas as pd

# Load the dataset
data = pd.read_csv("preprocess2024_test_1.csv")

# Convert columns 11 onward to numeric (coerce invalid values to NaN)
for column in data.columns[11:]:
    data[column] = pd.to_numeric(data[column], errors='coerce')

# Calculate column averages (ignoring NaN)
column_averages = data.iloc[:, 11:].mean()

# Fill missing values (NaN) with column averages
data.iloc[:, 11:] = data.iloc[:, 11:].fillna(column_averages)

unneeded_features = [
"id"    
,"home_team_abbr"
,"away_team_abbr"
,"is_night_game"
,"season"
,"home_team_rest"
,"away_team_rest"
,"home_pitcher_rest"
,"away_pitcher_rest"
,"home_pitcher"
,"away_pitcher"
,"home_team_season"
,"away_team_season"]

data.drop(unneeded_features, axis = 1,  inplace = True)
prefix = ["home_", "away_"]
suffix = ["_std", "_skew"]
Team_Seasonal = [
"team_errors"
,"team_spread"
,"team_wins"
,"batting_batting_avg"
,"batting_onbase_perc"
,"batting_onbase_plus_slugging"
,"batting_leverage_index_avg"
,"batting_wpa_bat"
,"batting_RBI"
,"pitching_earned_run_avg"
,"pitching_SO_batters_faced"
,"pitching_H_batters_faced"
,"pitching_BB_batters_faced"
,"pitching_leverage_index_avg"
,"pitching_wpa_def"]

for pre in prefix:
    for suf in suffix:
        for TS in Team_Seasonal:
            TS = pre + TS + suf
            data.drop(TS, axis = 1,  inplace = True)
prefix = ["home_", "away_"]
suffix = ["_std", "_skew"]
Pitcher_Seasonal = [
"pitcher_earned_run_avg"
,"pitcher_SO_batters_faced"
,"pitcher_H_batters_faced"
,"pitcher_BB_batters_faced"
,"pitcher_leverage_index_avg"
,"pitcher_wpa_def"]

for pre in prefix:
    for suf in suffix:
        for PS in Pitcher_Seasonal:
            PS = pre + PS + suf
            data.drop(PS, axis = 1,  inplace = True)

# Save the updated DataFrame to a new CSV file
data.to_csv("preprocess2024_test_2.csv", index=False)
