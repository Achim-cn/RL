#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.decomposition import PCA


pd.options.display.max_columns = 100


# In[2]:


feed_emb = pd.read_csv('/wbdc2021/data/wedata/wechat_algo_data2/feed_embeddings.csv')
emb_vec = feed_emb.feed_embedding.str.split(' ', expand=True)


# In[3]:


emb_vec = emb_vec.iloc[:, :512]
emb_vec.columns = [f'feed_emb_vec_{i}' for i in range(512)]


# In[4]:


pca_model = PCA(n_components=15)
emb_vec_pca = pca_model.fit_transform(emb_vec)
emb_vec_pca = pd.DataFrame(emb_vec_pca)
emb_vec_pca.columns = [f'feed_emb_vec_{i}' for i in range(15)]
emb_vec_pca['feedid'] = feed_emb.feedid.values


# In[5]:


features_df = emb_vec_pca.set_index('feedid').copy()


features_df.to_pickle('./proc/emb_vec_pca_20.pkl')


# In[ ]:





# In[6]:


train = pd.read_pickle('./proc/train.pkl')


# In[7]:


test_chusai = pd.read_pickle('./proc/chusai_test_df.pkl')
test_fusai = pd.read_pickle('./proc/fusai_test_a.pkl')

total_label = pd.concat([train, test_chusai, test_fusai], ignore_index=True)


# In[8]:


raw_feats = list(total_label)
total_label = total_label[['instance_id', 'feedid']]


# In[9]:


total_label = total_label.merge(features_df, how='left', on=['feedid'])


# In[10]:


features_df = total_label[['instance_id'] + [col for col in list(total_label) if col not in raw_feats]]


# In[11]:


features_df = features_df.set_index('instance_id')
features_df = reduce_mem_usage(features_df, list(features_df))
features_df.to_pickle('./features/tr_feedid_512_emb.pkl')




