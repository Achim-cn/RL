#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
# from gensim.models import Word2Vec

from tools import reduce_mem_usage
import numpy as np
import gc

pd.options.display.max_columns = 100


# In[2]:


train = pd.read_pickle('./proc/train.pkl')


# In[3]:


test_chusai = pd.read_pickle('./proc/chusai_test_df.pkl')
test_fusai = pd.read_pickle('./proc/fusai_test_a.pkl')

total_label = pd.concat([train, test_chusai, test_fusai], ignore_index=True)


# In[4]:


raw_feats = list(total_label)
total_label = total_label[['instance_id', 'userid', 'feedid', 'device', 'date_', 'authorid', 'bgm_song_id', 'bgm_singer_id']]


# In[5]:


total_label.head()


# In[6]:

total_label[f'today_total_cnt'] = total_label.groupby(['date_'])['instance_id'].transform('size')
total_label[f'feedid_today_cnt'] = total_label.groupby(['feedid', 'date_'])['instance_id'].transform('size')
total_label[f'feedid_today_cnt_rate'] = total_label[f'feedid_today_cnt'] / total_label['today_total_cnt']

total_label[f'authorid_today_cnt'] = total_label.groupby(['authorid', 'date_'])['instance_id'].transform('size')
total_label[f'authorid_today_cnt_rate'] = total_label[f'authorid_today_cnt'] / total_label['today_total_cnt']

total_label[f'bgm_song_id_today_cnt'] = total_label.groupby(['bgm_song_id', 'date_'])['instance_id'].transform('size')
total_label[f'bgm_song_id_today_cnt_rate'] = total_label[f'bgm_song_id_today_cnt'] / total_label['today_total_cnt']

total_label[f'bgm_singer_id_today_cnt'] = total_label.groupby(['bgm_singer_id', 'date_'])['instance_id'].transform('size')
total_label[f'bgm_singer_id_today_cnt_rate'] = total_label[f'bgm_singer_id_today_cnt'] / total_label['today_total_cnt']

total_label[f'feedid_today_cnt_of_authorid'] = total_label['feedid_today_cnt'] / total_label['authorid_today_cnt']

# author
total_label[f'authorid_today_feed_nunique'] = total_label.groupby(['authorid', 'date_'])['feedid'].transform('nunique')
total_label[f'authorid_today_song_nunique'] = total_label.groupby(['authorid', 'date_'])['bgm_song_id'].transform('nunique')
total_label[f'authorid_today_singer_nunique'] = total_label.groupby(['authorid', 'date_'])['bgm_singer_id'].transform('nunique')
total_label[f'authorid_today_user_nunique'] = total_label.groupby(['authorid', 'date_'])['userid'].transform('nunique')
total_label[f'authorid_today_cnt_by_user'] = total_label[f'authorid_today_cnt'] / total_label[f'authorid_today_user_nunique']
total_label = total_label.drop(['authorid_today_cnt'], axis=1)



total_label = total_label.drop(['today_total_cnt', 'feedid_today_cnt',
                                'bgm_song_id_today_cnt',
                                'bgm_singer_id_today_cnt'], axis=1)


# In[11]:


# # song_id
# total_label[f'song_today_feed_nunique'] = total_label.groupby(['bgm_song_id', 'date_'])['feedid'].transform('nunique')
# total_label[f'song_today_singer_nunique'] = total_label.groupby(['bgm_song_id', 'date_'])['bgm_singer_id'].transform('nunique')
# total_label[f'song_today_user_nunique'] = total_label.groupby(['bgm_song_id', 'date_'])['userid'].transform('nunique')

# # singer id
# total_label[f'singer_today_feed_nunique'] = total_label.groupby(['bgm_singer_id', 'date_'])['feedid'].transform('nunique')
# total_label[f'singer_today_song_nunique'] = total_label.groupby(['bgm_singer_id', 'date_'])['bgm_song_id'].transform('nunique')
# total_label[f'singer_today_user_nunique'] = total_label.groupby(['bgm_singer_id', 'date_'])['userid'].transform('nunique')


# In[18]:


# feed_info = pd.read_csv('../wbdc2021/data/wedata/wechat_algo_data2/feed_info.csv')


# In[19]:


# feed_info = feed_info[['feedid', 'machine_keyword_list', 'machine_tag_list']].copy()


# In[20]:


# for col in ['machine_tag_list']:
#     feed_info[col] = feed_info[col].fillna('-1').apply(lambda x: x.split(';')[0].split(' ')[0])


# In[21]:


# feed_info['machine_keyword_list'] = feed_info['machine_keyword_list'].fillna('-1').apply(lambda x: x.split(';')[0])


# In[23]:


# feed_info['machine_keyword_list'] = feed_info['machine_keyword_list'].replace('-1', np.nan)
# feed_info['machine_tag_list'] = feed_info['machine_tag_list'].replace('-1', np.nan)


# In[24]:


# total_label = total_label.merge(feed_info, how='left', on=['feedid'])


# In[25]:


# # list当天的流行度
# total_label[f'machine_tag_list_today_feed_nunique'] = total_label.groupby(['machine_tag_list', 'date_'])['feedid'].transform('nunique')
# total_label[f'machine_keyword_list_today_feed_nunique'] = total_label.groupby(['machine_keyword_list', 'date_'])['feedid'].transform('nunique')


# In[27]:


# # list当天的流行度
# total_label[f'machine_tag_list_today_feed_size'] = total_label.groupby(['machine_tag_list', 'date_'])['feedid'].transform('size')
# total_label[f'machine_keyword_list_today_feed_size'] = total_label.groupby(['machine_keyword_list', 'date_'])['feedid'].transform('size')


# In[28]:


# total_label[f'feedid_today_size'] = total_label.groupby(['feedid', 'date_'])['instance_id'].transform('size')


# In[29]:


# total_label[f'machine_tag_list_today_feed_size_rate'] = total_label[f'feedid_today_size'] / total_label[f'machine_tag_list_today_feed_size']
# total_label[f'machine_keyword_list_today_feed_size_rate'] = total_label[f'feedid_today_size'] / total_label[f'machine_keyword_list_today_feed_size']


# In[30]:


# del total_label[f'feedid_today_size']


# In[31]:


# # 用户list当天的次数
# total_label[f'user_machine_tag_list_today_size'] = total_label.groupby(['userid', 'machine_tag_list', 'date_'])['feedid'].transform('size')
# total_label[f'user_machine_keyword_list_today_size'] = total_label.groupby(['userid', 'machine_keyword_list', 'date_'])['feedid'].transform('size')

# # 用户当天的总的次数
# total_label[f'user_today_size'] = total_label.groupby(['userid', 'date_'])['feedid'].transform('size')

# total_label[f'user_machine_tag_list_today_size'] = total_label[f'user_machine_tag_list_today_size'] / total_label[f'user_today_size']
# total_label[f'user_machine_keyword_list_today_size'] = total_label[f'user_machine_keyword_list_today_size'] / total_label[f'user_today_size']


# In[7]:


features_df = total_label[['instance_id'] + [col for col in list(total_label) if col not in raw_feats + ['machine_keyword_list', 'machine_tag_list']]]


# In[8]:



features_df = features_df.set_index('instance_id')
features_df = reduce_mem_usage(features_df, list(features_df))
print(features_df.head())

features_df.to_pickle('./features/tr_item_id_today_cnt.pkl')


