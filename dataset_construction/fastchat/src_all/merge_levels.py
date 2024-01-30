import pandas as pd
from tqdm import tqdm

EG_data = pd.read_csv('./data_cleaned/EG.csv', sep='\t', dtype='string')

md52date = {}
for rowid, row in tqdm(EG_data.iterrows(), total=len(EG_data)):
    md52date[row['md5']] = row['date']

all_events = []

for y in range(2015, 2023):
    level1_df_y = pd.read_csv('./results/level1/{}_events.csv'.format(y), sep='\t', dtype='string')
    level2_df_y = pd.read_csv('./results/level2/{}_events.csv'.format(y), sep='\t', dtype='string')
    level3_df_y = pd.read_csv('./results/level3/{}_events.csv'.format(y), sep='\t', dtype='string')

    level1_only_events = level1_df_y[~level1_df_y.index.isin(level2_df_y['Level1_rowid'].astype(int))]
    level2_only_events = level2_df_y[~level2_df_y.index.isin(level3_df_y['Level3_rowid'].astype(int))]
    all_events_y = pd.concat([level1_only_events, level2_only_events, level3_df_y], ignore_index=True)
    all_events_y = all_events_y.drop(columns=['Level1_rowid', 'Level3_rowid'])
    all_events_y['Date'] = [int(md52date[md5]) for md5 in all_events_y['Md5']]
    all_events_y = all_events_y.sort_values(by=['Date'], ignore_index=True)
    all_events.append(all_events_y)

all_events_df = pd.concat(all_events, ignore_index=True)

all_events_df.to_csv(path_or_buf='./results/all_events_df_EG.csv', sep='\t', index=False)