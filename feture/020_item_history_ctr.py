#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from tqdm import tqdm

pd.options.display.max_columns = 100
train = pd.read_pickle('../data/train.pkl')
train.head()

for col in ['authorid', 'bgm_song_id', 'bgm_singer_id']:
    train[col] = train[col].fillna(-1)
raw_feats = list(train)
total_periods = sorted(train.date_.unique())
L = 10
window = 30
y_cols = ['comment', 'like', 'click_avatar', 'forward', 'follow', 'favorite', 'read_comment']
total_features = []
for ref_date in tqdm(total_periods):
    # 历史记录
    day_diff_mask = ref_date - train.date_
    history_log = train[(day_diff_mask <= window) & (day_diff_mask > 0)]
    tmp_label = train[train.date_ == ref_date].reset_index(drop=True).copy()

    for t in y_cols:
        
        if t == 'read_comment':
            history_log = history_log[history_log.device == 2].reset_index(drop=True)

        for grp in [['feedid'], ['authorid'], ['bgm_song_id'], ['bgm_singer_id']]:

            target_name = f'{"_".join(grp)}_{t}_smooth_mean_all'
            mn = history_log[t].mean()
            tmp = history_log.groupby(grp)[t].agg(['mean', 'count'])
            tmp['smooth'] = (tmp['mean'] * tmp['count'])
            tmp['smooth'] += (mn * L)
            tmp['smooth'] /= (tmp['count'] + L)
            tmp = tmp.rename(columns={'smooth': target_name})
            tmp = tmp.drop(['mean', 'count'], axis=1).reset_index()

            tmp_label = tmp_label.merge(tmp, how='left', on=grp)

    total_features.append(tmp_label)

total_features = pd.concat(total_features, ignore_index=True)
features_df = total_features[['instance_id'] + [col for col in list(total_features) if col not in raw_feats]]
features_df = features_df.set_index('instance_id')

features_df.to_pickle('../data/tr_item_hist_ctr.pkl')

for t in tqdm(y_cols):
    
    history_log = train.copy()
    
    # if t == 'read_comment':
    #     history_log = history_log[history_log.device == 2].reset_index(drop=True)
    
    for grp in [['feedid'], ['authorid'], ['bgm_song_id'], ['bgm_singer_id']]:
 
        target_name = f'{"_".join(grp)}_{t}_smooth_mean_all'
        mn = history_log[t].mean()
        tmp = history_log.groupby(grp)[t].agg(['mean', 'count'])
        tmp['smooth'] = (tmp['mean'] * tmp['count'])
        tmp['smooth'] += (mn * L)
        tmp['smooth'] /= (tmp['count'] + L)
        tmp = tmp.rename(columns={'smooth': target_name})
        tmp = tmp.drop(['mean', 'count'], axis=1)


        tmp.to_pickle(f'../features/tmp/{"_".join(grp)}_{t}.pkl')


from glob import glob

def concat_features(path):
    concat = []
    for file in sorted(glob(path)):
        print(file)
        concat.append(pd.read_pickle(file))
    print('concat all features !!!')
    concat = pd.concat(concat, axis=1)
    return concat

feedid = concat_features('../features/tmp/feedid*.pkl')
authorid = concat_features('../features/tmp/authorid*.pkl')
bgm_song_id = concat_features('../features/tmp/bgm_song_id*.pkl')
bgm_singer_id = concat_features('../features/tmp/bgm_singer_id*.pkl')

feedid.to_pickle('../features/te_feedid_hist_ctr.pkl')
authorid.to_pickle('../features/te_authorid_hist_ctr.pkl')
bgm_song_id.to_pickle('../features/te_bgm_song_id_hist_ctr.pkl')
bgm_singer_id.to_pickle('../features/te_bgm_singer_id_hist_ctr.pkl')


