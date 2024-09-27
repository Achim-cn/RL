#! -*- coding:utf-8 -*-
import os,gc
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# data_path = "/home/tione/notebook/wbdc2021-semi/"

if not os.path.exists("../data/embfeatures" ):
    os.mkdir("../data/embfeatures")
if not os.path.exists("../data/id2index" ):
    os.mkdir("../data/id2index")

# Data
feed_info = pd.read_csv('../data/feed_info.csv')
# user_action2 = pd.read_csv('../data/user_action.csv')
user_action = pd.read_csv('../data/user_action.csv')
# user_action = pd.concat([user_action1, user_action2], axis=0).reset_index(drop=True)
user_action = user_action.drop_duplicates(subset=["userid","feedid"],keep="last").reset_index(drop=True)

feed_info = feed_info[["feedid","authorid","machine_tag_list","manual_tag_list"]]

# feed info
feed_info[["feedid","authorid"]] = feed_info[["feedid","authorid"]].fillna(-1)
                      
def machine_tag(x):
    if len(x) < 1:
        return ''
    else:
        machine_tag_ = []
        word_weight = x.split(';')[:-1]
        for item in word_weight:
            tmp = item.split(" ")
            machine_tag_.append(tmp[0])
        return ";".join(machine_tag_)

feed_info['machine_tag_list'] = feed_info['machine_tag_list'].fillna('x')
feed_info['manual_tag_list'] = feed_info['manual_tag_list'].fillna('x')    
feed_info['machine_tag_list'] = feed_info['machine_tag_list'].fillna('x').apply(lambda x: machine_tag(str(x)))
feed_info['tag_list'] = feed_info['machine_tag_list']+';'+feed_info['manual_tag_list']
feed_info['tag_list'] = feed_info['tag_list'].apply(lambda x:x[2:] if x[0]=='x' else x)
feed_info['tag_list'] = feed_info['tag_list'].apply(lambda x:x[:-2] if x[-1]=='x' else x)
feed_info['tag_list'] = feed_info['tag_list'].apply(lambda x: str(x).replace(";"," ")) 

# 合并数据
data = user_action.drop_duplicates(subset=["userid","feedid"],keep="last").reset_index(drop=True)
data = data.merge(feed_info[["feedid","authorid","tag_list"]], on='feedid', how='left')
                      
del user_action
gc.collect()


dim_n = 32


## user-tag
if not os.path.exists("../data/embfeatures/user_tag_vec_{dim}.pickle".format(dim=str(dim_n))):
    data["tag_list"] = data["tag_list"].astype(str)
    user_tag = data.groupby("userid")["tag_list"].agg(list).reset_index()
    user_tag["tag_list"] = user_tag["tag_list"].apply(lambda x: " ".join(x))
    user_tag.columns = ["userid","userid_tag_list"]

    print("vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True,analyzer='word',ngram_range=(1, 1),min_df=3,max_df=0.94)
    tfidf_vectorizer.fit(user_tag["userid_tag_list"])
    tfidf_feat = tfidf_vectorizer.transform(user_tag["userid_tag_list"])

    print("TruncatedSVD...")
    svd = TruncatedSVD(n_components=dim_n, n_iter=5, random_state=2009)
    svd.fit(tfidf_feat)
    tfidf_svd = svd.transform(tfidf_feat)
    df_svd = pd.DataFrame(tfidf_svd, columns=["user_tags"+'_svd'+str(i) for i in range(dim_n)])
    df_feat = pd.concat([user_tag[["userid"]], df_svd], axis=1)
    df_feat.to_pickle("../data/embfeatures/user_tag_vec_{dim}.pickle".format(dim=str(dim_n)))

    del user_tag
    gc.collect()
else:
    print("../data/embfeatures/user_tag_vec_{dim}.pickle".format(dim=str(dim_n)))

# feed
if not os.path.exists("../data/embfeatures/feed_tag_vec_{dim}.pickle".format(dim=str(dim_n))):
    data["tag_list"] = data["tag_list"].astype(str)
    tmp = data.groupby("feedid")["tag_list"].agg(list).reset_index()
    tmp["tag_list"] = tmp["tag_list"].apply(lambda x: " ".join(x))
    tmp.columns = ["feedid","tag_tmp"]

    feed_info["tag_list"] = feed_info["tag_list"].astype(str)
    feed_tag = feed_info.groupby("feedid")["tag_list"].agg(list).reset_index()
    feed_tag["tag_list"] = feed_tag["tag_list"].apply(lambda x: " ".join(x))
    feed_tag.columns = ["feedid","tag"]
    feed_tag = feed_tag.merge(tmp,how='left',on="feedid").fillna('x')
    feed_tag["tag"] = feed_tag["tag"] + ' ' +feed_tag["tag_tmp"]

    print(tmp.shape)
    print(feed_tag.shape)

    del tmp, feed_tag["tag_tmp"]
    gc.collect()

    print("vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True,analyzer='word',ngram_range=(1, 1),min_df=3,max_df=0.94)
    tfidf_vectorizer.fit(feed_tag["tag"])
    tfidf_feat = tfidf_vectorizer.transform(feed_tag["tag"])

    print("TruncatedSVD...")
    svd = TruncatedSVD(n_components=dim_n, n_iter=5, random_state=2009)
    svd.fit(tfidf_feat)
    tfidf_svd = svd.transform(tfidf_feat)
    df_svd = pd.DataFrame(tfidf_svd, columns=["feed_tags"+'_svd'+str(i) for i in range(dim_n)])
    df_feat = pd.concat([feed_tag[["feedid"]], df_svd], axis=1)
    df_feat.to_pickle("../data/embfeatures/feed_tag_vec_{dim}.pickle".format(dim=str(dim_n)))
else:
    print("../data/embfeatures/feed_tag_vec_{dim}.pickle".format(dim=str(dim_n)))
    
# # author
# data["tag_list"] = data["tag_list"].astype(str)
# tmp = data.groupby("authorid")["tag_list"].agg(list).reset_index()
# tmp["tag_list"] = tmp["tag_list"].apply(lambda x: " ".join(x))
# tmp.columns = ["authorid","tag_tmp"]

# feed_info["tag_list"] = feed_info["tag_list"].astype(str)
# author_tag = feed_info.groupby("authorid")["tag_list"].agg(list).reset_index()
# author_tag["tag_list"] = author_tag["tag_list"].apply(lambda x: " ".join(x))
# author_tag.columns = ["authorid","tag"]
# author_tag = author_tag.merge(tmp,how='left',on="authorid").fillna('x')
# author_tag["tag"] = author_tag["tag"] + ' ' +author_tag["tag_tmp"]

# print(tmp.shape)
# print(author_tag.shape)

# del tmp, author_tag["tag_tmp"]
# gc.collect()

# print("vectorizer...")
# tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True,analyzer='word',ngram_range=(1, 1),min_df=3,max_df=0.94)
# tfidf_vectorizer.fit(author_tag["tag"])
# tfidf_feat = tfidf_vectorizer.transform(author_tag["tag"])

# print("TruncatedSVD...")
# svd = TruncatedSVD(n_components=dim_n, n_iter=5, random_state=2009)
# svd.fit(tfidf_feat)
# tfidf_svd = svd.transform(tfidf_feat)
# df_svd = pd.DataFrame(tfidf_svd, columns=["author_tags"+'_svd'+str(i) for i in range(dim_n)])
# df_feat = pd.concat([author_tag[["authorid"]], df_svd], axis=1)
# df_feat.to_pickle(data_path+"data/features/author_tag_vec_{dim}.pickle".format(dim=str(dim_n)))