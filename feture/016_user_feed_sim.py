#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gc

import numpy as np
import pandas as pd
from tqdm import tqdm

pd.options.display.max_columns = 100


# 余弦相似度
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim == 1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim == 2:
        a_norm = np.linalg.norm(a, axis=1)  # keepdims=True
        print(f"a_norm is {a_norm}")
        b_norm = np.linalg.norm(b, axis=1)  # keepdims=True
        print(f"b_norm is {b_norm}")
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similarity = np.dot(a, b.T)/(a_norm * b_norm)

    return similarity


# In[2]:


features_df = pd.read_pickle('../proc/emb_vec_pca_20.pkl')


# In[3]:


features_df = features_df.reset_index()


# In[4]:


features_df


# In[5]:


by = 'userid'
key = 'feedid'
w2v_nums = 15

agg_feats = [f'feed_emb_vec_' + str(i) for i in range(w2v_nums)]
y_cols = ['read_comment', 'comment', 'like', 'click_avatar', 'forward', 'follow', 'favorite']

mappers_dict = {}
cate_cols = ['feedid']


# In[6]:


train = pd.read_pickle('../proc/train.pkl')


# In[7]:


cate_offset = 0
for col in tqdm(cate_cols):
    cate2idx = {}
    for v in features_df[col].unique():
        if (v != v) | (v is None):
            continue
        cate2idx[v] = len(cate2idx) + cate_offset
    mappers_dict[col] = cate2idx
    features_df[f'{col}'] = features_df[col].map(cate2idx)
    train[f'{col}'] = train[col].map(cate2idx)
    cate_offset += len(cate2idx)


# In[8]:


y_cols = ['read_comment', 'comment', 'like', 'click_avatar', 'forward', 'follow', 'favorite']

train = train.merge(features_df, how='left', on=['feedid'])
raw_feats = list(train)


# In[9]:


train.head()


# In[10]:


item_feat = dict(zip(features_df['feedid'], features_df[agg_feats].values))


# In[11]:


item_n = len(item_feat)
item_np = np.zeros((item_n, 15))

for k, v in item_feat.items():
    item_np[k, :] = v

all_items = np.array(sorted(item_feat.keys()))

item_np = item_np/(np.linalg.norm(item_np, axis=1, keepdims=True)+1e-9)


# In[30]:


pd.to_pickle([item_np, mappers_dict], '../proc/feed_sim_file.pkl')


# In[ ]:





# In[35]:


total_periods = sorted(train.date_.unique())
window=30


# In[37]:


total_features = []
for ref_date in tqdm(total_periods):

    # 过去window天内的记录
    day_diff_mask = ref_date - train.date_
    history_log = train[(day_diff_mask <= window) & (day_diff_mask > 0)].copy()
    tmp_label = train[train.date_ == ref_date].reset_index(drop=True).copy()

    # 曝光相似性
    grp = history_log.groupby('userid')['feedid'].apply(lambda x: list(x)).to_frame('left_items_list').reset_index()
    
    tmp = tmp_label.merge(grp, how='inner', on=['userid'])
    tmp = tmp.rename(columns={'feedid': 'item'})

    feat = tmp[['instance_id', 'left_items_list', 'item']].copy()

    batch_size = 30000
    n = len(feat)
    batch_num = n // batch_size if n % batch_size == 0 else n // batch_size + 1

    feat['left_len'] = feat['left_items_list'].apply(len)
    feat_left = feat.sort_values('left_len')
    feat_left_len = feat_left['left_len'].values

    feat_left_items_list = feat_left['left_items_list'].values
    feat_left_items = feat_left['item'].values

    left_result = np.zeros((len(feat_left), 2))
    left_result_len = np.zeros(len(feat_left))

    len_max_nums = 300

    for i in range(batch_num):
        cur_batch_size = len(feat_left_len[i * batch_size:(i + 1) * batch_size])

        max_len = feat_left_len[i * batch_size:(i + 1) * batch_size].max()
        max_len = min(max(max_len, 1), len_max_nums)
        left_items = np.zeros((cur_batch_size, max_len), dtype='int32')
        for j, arr in enumerate(feat_left_items_list[i * batch_size:(i + 1) * batch_size]):
            arr = arr[:len_max_nums]
            left_items[j][:len(arr)] = arr

        left_result_len[i * batch_size:(i + 1) * batch_size] = np.isin(left_items, all_items).sum(axis=1)

        vec1 = item_np[left_items]
        vec2 = item_np[feat_left_items[i * batch_size:(i + 1) * batch_size]]
        vec2 = vec2.reshape(-1, 1, 15)
        sim = np.sum(vec1 * vec2, axis=-1)

        left_result[i * batch_size:(i + 1) * batch_size, 0] = sim.max(axis=1)
        left_result[i * batch_size:(i + 1) * batch_size, 1] = sim.sum(axis=1)

    #             if i % 10 == 0:
    #                 print('batch num',i)

    df_left = pd.DataFrame(left_result, index=feat_left.instance_id,
                           columns=['left_feedid_sim_max', 'left_feedid_sim_sum'])
    df_left['left_sim_len'] = left_result_len

    df_left[f'left_feedid_sim_mean'] = df_left[f'left_feedid_sim_sum'] / (
                df_left[f'left_sim_len'] + 1e-9)
    df_left = df_left.drop([f'left_feedid_sim_sum', f'left_sim_len'], axis=1)
    
    # 各个行为的相似性
    for y in y_cols:
        grp = history_log[history_log[y] == 1].groupby('userid')['feedid'].apply(lambda x: list(x)).to_frame(
            'left_items_list').reset_index()

        tmp = tmp_label.merge(grp, how='inner', on=['userid'])
        tmp = tmp.rename(columns={'feedid': 'item'})

        feat = tmp[['instance_id', 'left_items_list', 'item']].copy()

        batch_size = 30000

        n = len(feat)
        batch_num = n // batch_size if n % batch_size == 0 else n // batch_size + 1

        feat['left_len'] = feat['left_items_list'].apply(len)
        feat_left = feat.sort_values('left_len')
        feat_left_len = feat_left['left_len'].values

        feat_left_items_list = feat_left['left_items_list'].values
        feat_left_items = feat_left['item'].values

        left_result = np.zeros((len(feat_left), 2))
        left_result_len = np.zeros(len(feat_left))

        len_max_nums = 400

        for i in range(batch_num):
            cur_batch_size = len(feat_left_len[i * batch_size:(i + 1) * batch_size])

            max_len = feat_left_len[i * batch_size:(i + 1) * batch_size].max()
            max_len = min(max(max_len, 1), len_max_nums)
            left_items = np.zeros((cur_batch_size, max_len), dtype='int32')
            for j, arr in enumerate(feat_left_items_list[i * batch_size:(i + 1) * batch_size]):
                arr = arr[:len_max_nums]
                left_items[j][:len(arr)] = arr

            left_result_len[i * batch_size:(i + 1) * batch_size] = np.isin(left_items, all_items).sum(axis=1)

            vec1 = item_np[left_items]
            vec2 = item_np[feat_left_items[i * batch_size:(i + 1) * batch_size]]
            vec2 = vec2.reshape(-1, 1, 15)
            sim = np.sum(vec1 * vec2, axis=-1)

            left_result[i * batch_size:(i + 1) * batch_size, 0] = sim.max(axis=1)
            left_result[i * batch_size:(i + 1) * batch_size, 1] = sim.sum(axis=1)

        #                 if i % 10 == 0:
        #                     print('batch num',i)

        df_left_y = pd.DataFrame(left_result, index=feat_left.instance_id,
                                 columns=[f'left_feedid_{y}_sim_max', f'left_feedid_{y}_sim_sum'])
        df_left_y[f'left_{y}_sim_len'] = left_result_len
        df_left_y[f'left_feedid_{y}_sim_mean'] = df_left_y[f'left_feedid_{y}_sim_sum'] / (
                    df_left_y[f'left_{y}_sim_len'] + 1e-9)
        df_left_y = df_left_y.drop([f'left_feedid_{y}_sim_sum', f'left_{y}_sim_len'], axis=1)

        df_left = pd.concat([df_left, df_left_y], axis=1)

    total_features.append(df_left)

total_features = pd.concat(total_features)
    
    


# In[38]:


total_features


# In[42]:


features_df = train[['instance_id']].merge(total_features, how='left', on='instance_id')


# In[45]:


features_df = features_df.set_index(['instance_id'])


# In[46]:


features_df = reduce_mem_usage(features_df, list(features_df))


# In[48]:


features_df.columns = [f'{col}_all' for col in list(features_df)]
features_df.to_pickle('../features/tr_feedid_emb_sim_p30d.pkl')


# In[ ]:





# In[ ]:





# In[ ]:





# ### test

# In[25]:


y_cols = ['read_comment', 'comment', 'like', 'click_avatar', 'forward', 'follow', 'favorite']


# In[26]:


res = []
for y in y_cols:
    grp = train[train[y] == 1].groupby('userid')['feedid'].apply(lambda x: list(x)).to_frame(
        f'{y}_left_items_list')
    res.append(grp)


# In[27]:


grp = train.groupby('userid')['feedid'].apply(lambda x: list(x)).to_frame('left_items_list')


# In[28]:


res.append(grp)


# In[29]:


res = pd.concat(res, axis=1)


# In[32]:


res.to_pickle('../proc/feed_userid_hist_list.pkl')


# In[ ]:





# In[ ]:





# In[ ]:




