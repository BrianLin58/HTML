import pandas as pd
import numpy as np

# 示例数据（假设已有比赛数据表）
data = pd.read_csv("train_rename.csv")

# 转换日期格式
data['date'] = pd.to_datetime(data['date'])
data.head()



fliter1 = np.isnan(data["home_pitcher"])


# t1 = data[fliter1][["home_pitcher", "home_pitcher_rest", "home_team_abbr", "away_pitcher", "away_team_abbr", "date"]]
#print(t1)

for idx, row in data.iterrows():
    if(pd.isna(data.at[idx, 'home_pitcher'])):
        if not np.isnan(row['home_pitcher_rest']): days = int(row['home_pitcher_rest'])
        else: continue

        target_day = row['date'] - pd.Timedelta(days=days)
        abbr = row["home_team_abbr"]

        candidates = data[
            (data['date'] == target_day) &
            ((data["home_team_abbr"] == abbr) | (data["away_team_abbr"] == abbr))
        ]

        # 如果有匹配的记录，补全 `home_pitcher`
        if not candidates.empty:
            # print(f'before: {data.at[idx, "home_pitcher"]}\n')
            if not candidates["home_pitcher"].isna().all():
                data.at[idx, "home_pitcher"] = candidates["home_pitcher"].dropna().iloc[0]
            elif not candidates["away_pitcher"].isna().all():
                data.at[idx, "home_pitcher"] = candidates["away_pitcher"].dropna().iloc[0]


for idx, row in data.iterrows():
    if(pd.isna(data.at[idx, 'away_pitcher'])):
        if not np.isnan(row['away_pitcher_rest']): days = int(row['away_pitcher_rest'])
        else: continue

        target_day = row['date'] - pd.Timedelta(days=days)
        abbr = row["away_team_abbr"]


        candidates = data[
            (data['date'] == target_day) &
            ((data["home_team_abbr"] == abbr) | (data["away_team_abbr"] == abbr))
        ]

        # 如果有匹配的记录，补全 `home_pitcher`
        if not candidates.empty:
            # print(f'before: {data.at[idx, "home_pitcher"]}\n')
            if not candidates["home_pitcher"].isna().all():
                data.at[idx, "away_pitcher"] = candidates["home_pitcher"].dropna().iloc[0]
            elif not candidates["away_pitcher"].isna().all():
                data.at[idx, "away_pitcher"] = candidates["away_pitcher"].dropna().iloc[0]
            # print(f'after: {data.at[idx, "home_pitcher"]}\n')

data.to_csv("train_pitcher_filled.csv", index=False)
        
        # 使用布尔序列过滤数据
        # print(f'before{row["home_pitcher"]}')
        # for idx2, row2 in data.iterrows():
        #     if(row2['date'] == target_day and row2["home_team_abbr"] == abbr):
        #         data.loc[idx, "home_pitcher"] = row2["home_pitcher"]
        #         break
        #     elif(row2['date'] == target_day and row2["away_team_abbr"] == abbr):
        #         data.loc[idx, "home_pitcher"] = row2["home_pitcher"]
        #         break
        # print(f'after{row["home_pitcher"]}')
        # tg1 = data[f1][["home_pitcher", "home_pitcher_rest", "home_team_abbr", "date"]]
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