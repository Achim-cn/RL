#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from tqdm import tqdm
import gc

pd.options.display.max_columns = 100

train = pd.read_pickle('../data/train.pkl')
train.head()
feed_info = pd.read_csv('../data/feed_info.csv')
feed_info = feed_info[['feedid', 'manual_keyword_list']]
train = train.merge(feed_info, how='left', on=['feedid'])

for col in ['manual_keyword_list']:
    train[col] = train[col].fillna('-1').apply(lambda x: x.split(';'))
raw_feats = list(train)
y_cols = ['comment', 'like', 'click_avatar', 'forward', 'follow', 'favorite', 'read_comment']
train = train[['instance_id', 'userid', 'feedid', 'device', 'manual_keyword_list', 'date_'] + y_cols].copy()
L = 10
window = 30
y_cols = ['comment', 'like', 'click_avatar', 'forward', 'follow', 'favorite', 'read_comment']
key = ['userid', 'manual_keyword_list']
total_periods = sorted(train.date_.unique())

train.head()

total_features = []
for ref_date in tqdm(total_periods):
    # 历史记录
    day_diff_mask = ref_date - train.date_
    history_log = train[(day_diff_mask <= window) & (day_diff_mask > 0)]
    tmp_label = train[train.date_ == ref_date].reset_index(drop=True).copy()

    tmp_label = tmp_label.explode(key[-1]).reset_index(drop=True)
    history_log = history_log.explode(key[-1]).reset_index(drop=True)

    for t in y_cols:

        # if t == 'read_comment':
        #     history_log = history_log[history_log.device == 2].reset_index(drop=True)

        target_name = f'{"_".join(key)}_{t}_smooth_mean_all'
        mn = history_log[t].mean()
        tmp = history_log.groupby(key)[t].agg(['mean', 'count'])
        tmp['smooth'] = (tmp['mean'] * tmp['count'])
        tmp['smooth'] += (mn * L)
        tmp['smooth'] /= (tmp['count'] + L)
        tmp = tmp.rename(columns={'smooth': target_name})
        tmp = tmp.drop(['mean', 'count'], axis=1).reset_index()

        tmp_label = tmp_label.merge(tmp, how='left', on=key)

    features = [col for col in list(tmp_label) if col not in raw_feats]

    tmp_label = tmp_label.groupby(['instance_id'])[features].agg({'mean'})
    tmp_label.columns = [f'{i}_{j}' for i, j in tmp_label.columns]
    tmp_label = tmp_label.reset_index()

    total_features.append(tmp_label)

total_features = pd.concat(total_features, ignore_index=True)
total_features.head()
del tmp_label, history_log
gc.collect()
features_df = total_features[['instance_id'] + [col for col in list(total_features) if col not in raw_feats]]
features_df = features_df.set_index('instance_id')
features_df = features_df.sort_index()

features_df.to_pickle('../data/tr_user_manual_keyword_list_hist_ctr.pkl')

