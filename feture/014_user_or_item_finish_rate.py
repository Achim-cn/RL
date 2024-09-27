#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from tqdm import tqdm


pd.options.display.max_columns = 100


# In[2]:


train = pd.read_pickle('./proc/train.pkl')


# In[3]:


train.head()


# In[4]:


total_periods = sorted(train.date_.unique())


# In[5]:


raw_feats = list(train)


# In[6]:


print(total_periods)


# In[7]:


train['play'] = train['play'] / 1000
train['stay'] = train['stay'] / 1000

train['play_div_stay'] = train['play'] / train['stay']

train['play_rate'] = train['play'] / train['videoplayseconds']
train['stay_rate'] = train['stay'] / train['videoplayseconds']


# In[8]:


window = 30


# In[10]:


total_features = []
for ref_date in tqdm(total_periods):

    # 过去window天内的记录
    day_diff_mask = ref_date - train.date_
    history_log = train[(day_diff_mask <= window) & (day_diff_mask > 0)]
    tmp_label = train[train.date_ == ref_date].reset_index(drop=True).copy()

    for grp in [['userid'],
                ['userid', 'authorid'],
                ['feedid'],
                ['authorid'],
                ]:
        tmp = history_log.groupby(grp)[['play', 'stay', 'play_div_stay', 'play_rate', 'stay_rate']].agg(['mean', 'std'])
        tmp.columns = [f'{"_".join(grp)}_{i}_{j}_past_all' for i, j in tmp.columns]
        tmp_label = tmp_label.merge(tmp, how='left', on=grp)

        total_features.append(tmp_label)

total_features = pd.concat(total_features, ignore_index=True)


# In[16]:


features_df = total_features[['instance_id'] + [col for col in list(total_features) if 'past_all' in col]]


features_df = features_df.set_index('instance_id')


features_df.tail()

features_df.to_pickle('./features/tr_user_or_feed_hist_finish_rate.pkl')


features_df.to_pickle('./features/tr_user_or_feed_hist_finish_rate.pkl')




