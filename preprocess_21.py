# do this after preprocess_2.py

import pandas as pd
import numpy as np

df = pd.read_csv("preprocess_2.csv")


# 假設數據框名稱為 df
def fill_missing_mean1(df):
    # 建立非NaN的mean-skew對應表
    mapping = df.dropna(subset=['home_team_wins_mean', 'home_team_wins_skew']).set_index('home_team_wins_skew')['home_team_wins_mean'].to_dict()
    
    # 填補mean為NaN的值
    df['home_team_wins_mean'] = df.apply(
        lambda row: mapping[row['home_team_wins_skew']] if pd.isna(row['home_team_wins_mean']) and row['home_team_wins_skew'] in mapping else row['home_team_wins_mean'],
        axis=1
    )
    df['away_team_wins_mean'] = df.apply(
        lambda row: mapping[row['away_team_wins_skew']] if pd.isna(row['away_team_wins_mean']) and row['away_team_wins_skew'] in mapping else row['away_team_wins_mean'],
        axis=1
    )
    return df


def fill_missing_mean2(df):
    # 建立非NaN的mean-skew對應表
    mapping = df.dropna(subset=['away_team_wins_mean', 'away_team_wins_skew']).set_index('away_team_wins_skew')['away_team_wins_mean'].to_dict()
    
    # 填補mean為NaN的值
    df['home_team_wins_mean'] = df.apply(
        lambda row: mapping[row['home_team_wins_skew']] if pd.isna(row['home_team_wins_mean']) and row['home_team_wins_skew'] in mapping else row['home_team_wins_mean'],
        axis=1
    )
    df['away_team_wins_mean'] = df.apply(
        lambda row: mapping[row['away_team_wins_skew']] if pd.isna(row['away_team_wins_mean']) and row['away_team_wins_skew'] in mapping else row['away_team_wins_mean'],
        axis=1
    )
    return df

# 使用函數處理數據
df = fill_missing_mean1(df)
df = fill_missing_mean2(df)
# 假設數據框名稱為 df
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['date', 'is_night_game'], ascending=[True, True])
df.reset_index(drop=True, inplace=True)
df['id'] = range(1, len(df) + 1)


team_last_game = np.empty(30, dtype=object)

# 例子：初始化每個位置為一個空字典
for i in range(30):
    team_last_game[i] = {'is_home_team': False, 'last_game_idx': None, 'last_game_wins_mean': None}

for idx, row in df.iterrows():
    home_abbr = row['home_team_abbr']
    away_abbr = row['away_team_abbr']
    # print(idx)
    # new season
    if idx == 0 or df.at[idx-1, 'season'] != row['season']:
        for i in range(30):
            team_last_game[i] = {'is_home_team': False, 'last_game_idx': None, 'last_game_wins_mean': None}
    
    if team_last_game[home_abbr]['last_game_idx'] == None:  # team's first game of season
        team_last_game[home_abbr] = {'is_home_team': True, 'last_game_idx': idx, 'last_game_wins_mean': None}
        
    elif not pd.isna(team_last_game[home_abbr]['last_game_wins_mean']):
    # teams' 2nd game of season
        if team_last_game[home_abbr]['last_game_wins_mean'] == None or (row['home_team_wins_mean'] == team_last_game[home_abbr]['last_game_wins_mean']) or (abs(row['home_team_wins_mean']) > 4):
            if (row['home_team_wins_mean'] > 0) ^ team_last_game[home_abbr]['is_home_team']:
                df.at[team_last_game[home_abbr]['last_game_idx'], 'home_team_win'] = 'False'
            else: 
                df.at[team_last_game[home_abbr]['last_game_idx'], 'home_team_win'] = 'True'

        elif row['home_team_wins_mean'] > team_last_game[home_abbr]['last_game_wins_mean']:
            if team_last_game[home_abbr]['is_home_team']:
                df.at[team_last_game[home_abbr]['last_game_idx'], 'home_team_win'] = 'True'
            else: 
                df.at[team_last_game[home_abbr]['last_game_idx'], 'home_team_win'] = 'False'
        
        else:
            if team_last_game[home_abbr]['is_home_team']:
                df.at[team_last_game[home_abbr]['last_game_idx'], 'home_team_win'] = 'False'
            else: 
                df.at[team_last_game[home_abbr]['last_game_idx'], 'home_team_win'] = 'True'

    if team_last_game[away_abbr]['last_game_idx'] == None:  # team's first game of season
        team_last_game[away_abbr] = {'is_home_team': False, 'last_game_idx': idx, 'last_game_wins_mean': None}

    elif not pd.isna(team_last_game[away_abbr]['last_game_wins_mean']):
        if team_last_game[away_abbr]['last_game_wins_mean'] == None or (row['away_team_wins_mean'] == team_last_game[away_abbr]['last_game_wins_mean'])or (abs(row['away_team_wins_mean']) > 4):
            if (row['away_team_wins_mean'] > 0) ^ team_last_game[away_abbr]['is_home_team']:
                df.at[team_last_game[away_abbr]['last_game_idx'], 'home_team_win'] = 'False'
            else: 
                df.at[team_last_game[away_abbr]['last_game_idx'], 'home_team_win'] = 'True'

        elif row['away_team_wins_mean'] > team_last_game[away_abbr]['last_game_wins_mean']:
            if team_last_game[away_abbr]['is_home_team']:
                df.at[team_last_game[away_abbr]['last_game_idx'], 'home_team_win'] = 'True'
            else: 
                df.at[team_last_game[away_abbr]['last_game_idx'], 'home_team_win'] = 'False'
        
        else:
            if team_last_game[away_abbr]['is_home_team']:
                df.at[team_last_game[away_abbr]['last_game_idx'], 'home_team_win'] = 'False'
            else: 
                df.at[team_last_game[away_abbr]['last_game_idx'], 'home_team_win'] = 'True'
    
    team_last_game[home_abbr] = {'is_home_team': True, 'last_game_idx': idx, 'last_game_wins_mean': row['home_team_wins_mean']}
    team_last_game[away_abbr] = {'is_home_team': False, 'last_game_idx': idx, 'last_game_wins_mean': row['away_team_wins_mean']}








df.to_csv('preprocess_2_with_true_label.csv', index=False)
# mean_list = []
# skew_list = []

# prepare both map from mean to skew and skew to mean 

# fill the cell if one of mean or skew is nan

# sort with year and date;

# prepare 30 float pre[30] to store the winrate of last game of each team

# for each game except first game of each season:
#     if pre[home] == nan (2ndgame)
#         if home_win_mean != nan
#             pre[home] = home_win_mean
#             if home_win_mean > 0
#                 home_win = true 
#             else home_win = false
#             continue
#     if pre[away] == nan (2ndgame)
#         if away_win_mean != nan 
#             pre[away] = home_win_mean
#             if away_win_mean > 0
#             home_win = true 
#             continue

#     if home_win_mean != nan
