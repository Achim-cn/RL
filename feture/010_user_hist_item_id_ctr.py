#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from tqdm import tqdm


pd.options.display.max_columns = 100


train = pd.read_pickle('data/train.pkl')

# 统计train数据的总数
total_count = len(train)

# 统计train中instance_id的这一列不同值总数
unique_instance_id_count = train['instance_id'].nunique()

print("总数:", total_count)
print("instance_id的不同值总数:", unique_instance_id_count)


for col in ['authorid', 'bgm_song_id', 'bgm_singer_id']:    # 几个id不共享embedding 编码映射， 所以可以都用-1
    train[col] = train[col].fillna(-1)

total_periods = sorted(train.date_.unique())

raw_feats = list(train)

print(total_periods)
# CTR贝叶斯平滑 ： 历史每天的
L = 10
window = 30
y_cols = ['comment', 'like', 'click_avatar', 'forward', 'follow', 'favorite', 'read_comment']


total_features = []
for ref_date in tqdm(total_periods):

    # 获取历史
    day_diff_mask = ref_date - train.date_ # ref_data=7,  data=1
    history_log = train[(day_diff_mask <= window) & (day_diff_mask > 0)]
    tmp_label = train[train.date_ == ref_date].reset_index(drop=True).copy()

    for t in y_cols:
        
        # if t == 'read_comment':
        #     history_log = history_log[history_log.device == 2].reset_index(drop=True)

        for grp in [['userid'],
                    ['userid', 'authorid'],
                    ['userid', 'bgm_song_id'],
                    ['userid', 'bgm_singer_id']
                    ]:

            target_name = f'{"_".join(grp)}_{t}_smooth_mean_all'
            mn = history_log[t].mean()
            tmp = history_log.groupby(grp)[t].agg(['mean', 'count'])
            tmp['smooth'] = (tmp['mean'] * tmp['count'])
            tmp['smooth'] += (mn * L)
            tmp['smooth'] /= (tmp['count'] + L)
            tmp = tmp.rename(columns={'smooth': target_name})
            tmp = tmp.drop(['mean', 'count'], axis=1).reset_index()

            tmp_label = tmp_label.merge(tmp, how='left', on=grp)
            # print(tmp_label)

    total_features.append(tmp_label)

total_features = pd.concat(total_features, ignore_index=True)


features_df = total_features[['instance_id'] + [col for col in list(total_features) if col not in raw_feats]]


features_df = features_df.set_index('instance_id')


features_df.head()
print(features_df)

features_df.to_pickle('../data/tr_user_hist_ctr.pkl')





