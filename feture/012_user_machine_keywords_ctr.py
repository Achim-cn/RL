#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from tqdm import tqdm
import gc


pd.options.display.max_columns = 100


# In[2]:


train = pd.read_pickle('./proc/train.pkl')


# In[3]:


train.head()


# In[4]:


feed_info = pd.read_csv('/wbdc2021/data/wedata/wechat_algo_data2/feed_info.csv')


# In[5]:


feed_info = feed_info[['feedid', 'machine_keyword_list']]
train = train.merge(feed_info, how='left', on=['feedid'])


# In[6]:


for col in ['machine_keyword_list']:
    train[col] = train[col].fillna('-1').apply(lambda x: x.split(';'))


# In[7]:


raw_feats = list(train)


# In[8]:


y_cols = ['comment', 'like', 'click_avatar', 'forward', 'follow', 'favorite', 'read_comment']


# In[9]:


train = train[['instance_id', 'userid', 'feedid', 'device', 'machine_keyword_list', 'date_'] + y_cols].copy()


# In[10]:


L = 10
window = 30
y_cols = ['comment', 'like', 'click_avatar', 'forward', 'follow', 'favorite', 'read_comment']


# In[11]:


key = ['userid', 'machine_keyword_list']


# In[12]:


total_periods = sorted(train.date_.unique())



train.head()



total_features = []
for ref_date in tqdm(total_periods):

    # 过去window天内的记录
    day_diff_mask = ref_date - train.date_
    history_log = train[(day_diff_mask <= window) & (day_diff_mask > 0)]
    tmp_label = train[train.date_ == ref_date].reset_index(drop=True).copy()

    tmp_label = tmp_label.explode(key[-1]).reset_index(drop=True)
    history_log = history_log.explode(key[-1]).reset_index(drop=True)

    for t in y_cols:

        if t == 'read_comment':
            history_log = history_log[history_log.device == 2].reset_index(drop=True)

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



features_df = total_features[['instance_id'] + [col for col in list(total_features) if col not in raw_feats]]



features_df = features_df.set_index('instance_id')



features_df = features_df.sort_index()



features_df.to_pickle('./features/tr_user_machine_keyword_list_hist_ctr.pkl')





