import pandas as pd
import numpy as np

# 示例数据（假设已有比赛数据表）
data = pd.read_csv("same_season_test_data.csv")


for idx, row in data.iterrows():
    if pd.isna(row["season"]):
        if not pd.isna(row["home_team_season"]):
            data.loc[idx, "season"] = row["home_team_season"][4:8]
        elif not pd.isna(row["away_team_season"]):
            data.loc[idx, "season"] = row["away_team_season"][4:8]

data.to_csv('newtest.csv')
print('finish')



# fliter1 = np.isnan(data["home_pitcher"])


# t1 = data[fliter1][["home_pitcher", "home_pitcher_rest", "home_team_abbr", "date"]]
# print(t1)

# for idx, row in t1.iterrows():
#     if ~np.isnan(row['home_pitcher_rest']): days = int(row['home_pitcher_rest'])

#     target_day = row['date'] - pd.Timedelta(days=days)
#     f1 = data['date'] == target_day
#     f2 = data["home_team_abbr"] == row["home_team_abbr"]
#     f3 = data["home_pitcher"] == row["home_pitcher"]
#     f4 = data["away_team_abbr"] == row["home_team_abbr"]
#     f5 = data["away_pitcher"] == row["home_pitcher"]
#     tg = data[f1 & (f2&f3 | f4&f5)][["home_pitcher", "home_pitcher_rest", "home_team_abbr", "date"]]
    #print(tg)
    #row["home_pitcher"] = tg[0]['home_pitcher']


# fliter1 = np.isnan(data["home_pitcher"])

# fliter1 = data["home_team_abbr"] == 0

# filter2 = data["away_team_abbr"] == 0
# filter3 = data["season"] == 2016

# "id", "home_team_abbr", "away_team_abbr", "date", 

# t1 = data[((fliter1 | filter2) & filter3)][["home_pitcher", "away_pitcher"]]
# l1 = np.unique(t1)
# print(l1)
# temp = temp.sort_values("date").reset_index()
# for idx, row in temp.iterrows():
#     print(type(row["home_pitcher"]))
#     if type(row["home_pitcher"]) != 'float':
#         print(row["id"])
# 补全 home_pitcher
# def fill_home_pitcher(data):
#     # 创建一个长表格，包含 home 和 away 的比赛记录
#     all_pitchers = pd.DataFrame({
#         'team_abbr': data['home_team_abbr'].tolist() + data['away_team_abbr'].tolist(),
#         'pitcher': data['home_pitcher'].tolist() + data['away_pitcher'].tolist(),
#         'date': data['date'].tolist() + data['date'].tolist(),
#         'is_home': [True] * len(data) + [False] * len(data)
#     })
    
#     # 按照队伍分组，记录每位投手的比赛日期
#     all_pitchers = all_pitchers.dropna(subset=['pitcher'])
#     all_pitchers.sort_values(by=['team_abbr', 'date'], inplace=True)
    
#     # 补全 home_pitcher
#     for idx, row in data.iterrows():
#         team = row['home_team_abbr']
#         date = row['date']
#         rest_days = row['home_pitcher_rest']
        
#         # 找到符合条件的最近投手
#         candidates = all_pitchers[
#             (all_pitchers['team_abbr'] == team) & 
#             (all_pitchers['date'] <= date)
#         ]
#         if not candidates.empty:
#             candidates['rest_diff'] = (date - candidates['date']).dt.days
#             closest_pitcher = candidates[
#                 candidates['rest_diff'] == rest_days
#             ]
#             if not closest_pitcher.empty:
#                 data.at[idx, 'home_pitcher'] = closest_pitcher.iloc[0]['pitcher']
#     return data

# # 填充数据
# filled_data = fill_home_pitcher(data)

# # 查看结果
# print(filled_data)