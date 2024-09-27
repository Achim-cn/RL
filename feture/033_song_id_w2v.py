#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from gensim.models import Word2Vec

from tools import reduce_mem_usage

import gc

pd.options.display.max_columns = 100


# In[6]:


train = pd.read_pickle('./proc/train.pkl')


# In[7]:


test_chusai = pd.read_pickle('./proc/chusai_test_df.pkl')
test_fusai = pd.read_pickle('./proc/fusai_test_a.pkl')

total_label = pd.concat([train, test_chusai, test_fusai], ignore_index=True)


# In[8]:


raw_feats = list(total_label)
total_label = total_label[['instance_id', 'userid', 'feedid', 'device', 'date_', 'authorid', 'bgm_song_id', 'bgm_singer_id']]


# In[9]:


for col in ['authorid', 'bgm_song_id', 'bgm_singer_id']:
    total_label[col] = total_label[col].fillna(-1)


# In[10]:


total_label.head()


# In[11]:


del train, test_chusai, test_fusai
gc.collect()


# In[ ]:





# In[12]:


by = 'userid'
key = 'bgm_song_id'
w2v_nums = 10
merge_vec = [f'w2v_{by}_{key}_' + str(i) for i in range(w2v_nums)]

df_user_ad_id_list = total_label.groupby(by).apply(lambda x: list(x[key])).reset_index().rename(columns={0: f'{key}_list'})


# In[13]:


key_list = list(df_user_ad_id_list[f'{key}_list'].values)
model = Word2Vec(key_list, vector_size=w2v_nums, window=5, min_count=1,
                 workers=12, seed=9482, sg=1, epochs=5)
print('Model Done ...')

vocab = list(model.wv.index_to_key)
w2v_arr = []

for v in vocab:
    w2v_arr.append(list(model.wv[v]))


# In[14]:


df_w2v = pd.DataFrame()
df_w2v[key] = vocab
df_w2v = pd.concat([df_w2v, pd.DataFrame(w2v_arr)], axis=1)
df_w2v.columns = [key] + [f'w2v_{by}_{key}_' + str(i) for i in range(w2v_nums)]

w2v_cols = [f'w2v_{by}_{key}_' + str(i) for i in range(w2v_nums)]


# In[ ]:





# In[15]:


features_df = df_w2v.set_index('bgm_song_id')
features_df = reduce_mem_usage(features_df, list(features_df))

features_df.to_pickle('./proc/all_bgm_song_id_w2v.pkl')


# In[ ]:





# In[ ]:





# In[16]:


# 构造线下特征
print('creating train features ...')
total_label = total_label.merge(features_df, on=key, how='left')
features_df = total_label[['instance_id'] + [col for col in list(total_label) if col not in raw_feats]]
del total_label


# In[17]:


features_df = features_df.set_index('instance_id')
features_df = reduce_mem_usage(features_df, list(features_df))

features_df.to_pickle('./features/tr_bgm_song_id_w2v.pkl')


# In[ ]:




