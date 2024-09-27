#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from gensim.models import Word2Vec

from tools import reduce_mem_usage

import gc

pd.options.display.max_columns = 100


# In[2]:


train = pd.read_pickle('./proc/train.pkl')


# In[3]:


test_chusai = pd.read_pickle('./proc/chusai_test_df.pkl')
test_fusai = pd.read_pickle('./proc/fusai_test_a.pkl')

total_label = pd.concat([train, test_chusai, test_fusai], ignore_index=True)


# In[4]:


del train, test_chusai, test_fusai
gc.collect()


# In[5]:


feed_info = pd.read_csv('/wbdc2021/data/wedata/wechat_algo_data2/feed_info.csv')


# In[6]:


feed_info = feed_info[['feedid', 'manual_keyword_list']]
total_label = total_label.merge(feed_info, how='left', on=['feedid'])


# In[7]:


for col in ['manual_keyword_list']:
    total_label[col] = total_label[col].fillna('-1').apply(lambda x: x.split(';'))


# In[8]:


total_label = total_label[['instance_id', 'userid', 'manual_keyword_list']]


# In[9]:


total_label.head()


# In[10]:


total_label = total_label.explode('manual_keyword_list').reset_index(drop=True)


# In[ ]:





# In[11]:


by = 'userid'
key = 'manual_keyword_list'
w2v_nums = 10
merge_vec = [f'w2v_{by}_{key}_' + str(i) for i in range(w2v_nums)]

df_user_ad_id_list = total_label.groupby(by).apply(lambda x: list(x[key])).reset_index().rename(columns={0: f'{key}_list'})


# In[12]:


key_list = list(df_user_ad_id_list[f'{key}_list'].values)
model = Word2Vec(key_list, vector_size=w2v_nums, window=5, min_count=1,
                 workers=48, seed=9482, sg=1, epochs=5)
print('Model Done ...')

vocab = list(model.wv.index_to_key)
w2v_arr = []

for v in vocab:
    w2v_arr.append(list(model.wv[v]))


# In[13]:


df_w2v = pd.DataFrame()
df_w2v[key] = vocab
df_w2v = pd.concat([df_w2v, pd.DataFrame(w2v_arr)], axis=1)
df_w2v.columns = [key] + [f'w2v_{by}_{key}_' + str(i) for i in range(w2v_nums)]

w2v_cols = [f'w2v_{by}_{key}_' + str(i) for i in range(w2v_nums)]


# In[14]:


features_df = df_w2v.set_index('manual_keyword_list')
features_df = reduce_mem_usage(features_df, list(features_df))

features_df.to_pickle('./proc/all_manual_keyword_list_w2v.pkl')


# In[ ]:





# In[15]:


# 构造线下特征
print('creating train features ...')
total_label = total_label.merge(features_df, on=key, how='left')

total_label = total_label.groupby(['instance_id'])[w2v_cols].agg({'mean'})
total_label.columns = [f'{i}_{j}' for i, j in total_label.columns]


# In[21]:


# features_df = total_label[[col for col in list(total_label) if col not in raw_feats]]
# features_df = features_df.set_index('instance_id')
total_label = reduce_mem_usage(total_label, list(total_label))
total_label.to_pickle('./features/tr_manual_keywords_w2v.pkl')


# In[ ]:




