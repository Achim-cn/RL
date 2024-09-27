#! -*- coding:utf-8 -*-
import os,gc
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

if not os.path.exists("data/embfeatures"):
    os.mkdir("data/embfeatures")
if not os.path.exists("data/id2index"):
    os.mkdir("data/id2index")

feed_info = pd.read_csv('data/feed_info.csv')
user_action = pd.read_csv('data/user_action.csv')
test = pd.read_csv('data/test_a.csv')


data = user_action[["userid","feedid"]].drop_duplicates(subset=["userid","feedid"],keep="last").reset_index(drop=True)
# test = test[["userid","feedid"]]
# data = pd.concat([data, test], axis=0).reset_index(drop=True)
data = data.merge(feed_info[["feedid","authorid"]], how='left',on="feedid")

dim_n = 150

# user_feed
if not os.path.exists("data/embfeatures/user_feed_vec_{dim}.pickle".format(dim=str(dim_n))):
    data["feedid"] = data["feedid"].astype(str)
    user_feed = data.groupby("userid")["feedid"].agg(list).reset_index()
    data["feedid"] = data["feedid"].astype(int)
    user_feed["feedid"] = user_feed["feedid"].apply(lambda x: " ".join(x))
    user_feed.columns = ["userid","userid_feedid_seqs"]

    print("vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True,analyzer='word',ngram_range=(1, 1),min_df=3,max_df=0.94)
    tfidf_vectorizer.fit(user_feed["userid_feedid_seqs"])
    tfidf_feat = tfidf_vectorizer.transform(user_feed["userid_feedid_seqs"])

    print("TruncatedSVD...")
    svd = TruncatedSVD(n_components=dim_n, n_iter=5, random_state=2009)
    svd.fit(tfidf_feat)
    tfidf_svd = svd.transform(tfidf_feat)
    df_svd = pd.DataFrame(tfidf_svd, columns=["user_feed"+'_svd'+str(i) for i in range(dim_n)])
    df_feat = pd.concat([user_feed[["userid"]], df_svd], axis=1)
    df_feat.to_pickle("data/embfeatures/user_feed_vec_{dim}.pickle".format(dim=str(dim_n)))

else:
    print("data/embfeatures/user_feed_vec_{dim}.pickle".format(dim=str(dim_n)))
    

# user_author
if not os.path.exists("data/embfeatures/user_author_vec_{dim}.pickle".format(dim=str(dim_n))):
    data["authorid"] = data["authorid"].astype(str)
    user_author = data.groupby("userid")["authorid"].agg(list).reset_index()
    data["authorid"] = data["authorid"].astype(int)
    user_author["authorid"] = user_author["authorid"].apply(lambda x: " ".join(x))
    user_author.columns = ["userid","userid_authorid_seqs"]

    print("vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True,analyzer='word',ngram_range=(1, 1),min_df=3,max_df=0.94)
    tfidf_vectorizer.fit(user_author["userid_authorid_seqs"])
    tfidf_feat = tfidf_vectorizer.transform(user_author["userid_authorid_seqs"])

    print("TruncatedSVD...")
    svd = TruncatedSVD(n_components=dim_n, n_iter=5, random_state=2009)
    svd.fit(tfidf_feat)
    tfidf_svd = svd.transform(tfidf_feat)
    df_svd = pd.DataFrame(tfidf_svd, columns=["user_author"+'_svd'+str(i) for i in range(dim_n)])
    df_feat = pd.concat([user_author[["userid"]], df_svd], axis=1)
    df_feat.to_pickle("data/embfeatures/user_author_vec_{dim}.pickle".format(dim=str(dim_n)))

    del user_author
    gc.collect()
else:
    print("data/embfeatures/user_author_vec_{dim}.pickle".format(dim=str(dim_n)))
    
# feed_user
if not os.path.exists("data/embfeatures/feed_user_vec_{dim}.pickle".format(dim=str(dim_n))):
    data["userid"] = data["userid"].astype(str)
    feed_user = data.groupby("feedid")["userid"].agg(list).reset_index()
    data["userid"] = data["userid"].astype(int)
    feed_user["userid"] = feed_user["userid"].apply(lambda x: " ".join(x))
    feed_user.columns = ["feedid","feedid_userid_seqs"]

    print("vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True,analyzer='word',ngram_range=(1, 1),min_df=3,max_df=0.94)
    tfidf_vectorizer.fit(feed_user["feedid_userid_seqs"])
    tfidf_feat = tfidf_vectorizer.transform(feed_user["feedid_userid_seqs"])

    print("TruncatedSVD...")
    svd = TruncatedSVD(n_components=dim_n, n_iter=5, random_state=2009)
    svd.fit(tfidf_feat)
    tfidf_svd = svd.transform(tfidf_feat)
    df_svd = pd.DataFrame(tfidf_svd, columns=["feed_user"+'_svd'+str(i) for i in range(dim_n)])
    df_feat = pd.concat([feed_user[["feedid"]], df_svd], axis=1)
    df_feat.to_pickle("data/embfeatures/feed_user_vec_{dim}.pickle".format(dim=str(dim_n)))

    del feed_user
    gc.collect()
else:
    print("data/embfeatures/feed_user_vec_{dim}.pickle".format(dim=str(dim_n)))

# author_user
if not os.path.exists("data/embfeatures/author_user_vec_{dim}.pickle".format(dim=str(dim_n))):
    data["userid"] = data["userid"].astype(str)
    author_user = data.groupby("authorid")["userid"].agg(list).reset_index()
    data["userid"] = data["userid"].astype(int)
    author_user["userid"] = author_user["userid"].apply(lambda x: " ".join(x))
    author_user.columns = ["authorid","authorid_userid_seqs"]

    print("vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True,analyzer='word',ngram_range=(1, 1),min_df=3,max_df=0.94)
    tfidf_vectorizer.fit(author_user["authorid_userid_seqs"])
    tfidf_feat = tfidf_vectorizer.transform(author_user["authorid_userid_seqs"])

    print("TruncatedSVD...")
    svd = TruncatedSVD(n_components=dim_n, n_iter=5, random_state=2009)
    svd.fit(tfidf_feat)
    tfidf_svd = svd.transform(tfidf_feat)
    df_svd = pd.DataFrame(tfidf_svd, columns=["author_user"+'_svd'+str(i) for i in range(dim_n)])
    df_feat = pd.concat([author_user[["authorid"]], df_svd], axis=1)
    df_feat.to_pickle("data/embfeatures/author_user_vec_{dim}.pickle".format(dim=str(dim_n)))

    del author_user
    gc.collect()
else:
    print("data/embfeatures/author_user_vec_{dim}.pickle".format(dim=str(dim_n)))

del feed_info,user_action,test,data
gc.collect()

if not os.path.exists("data/embfeatures/feed_emb_vec_512.pickle"):
    feed_embeddings = pd.read_csv('data/feed_embeddings.csv')
    feed_embeddings['feed_embedding'] = feed_embeddings['feed_embedding'].apply(lambda x:x.split(" ")[:-1])

    emb = feed_embeddings['feed_embedding'].apply(pd.Series,index=['e'+str(i) for i in range(512)]).reset_index(drop=True)
    emb = emb.astype(float)

    df_emb = pd.concat([feed_embeddings[['feedid']], emb], axis=1)
    print("to csv...")
    df_emb.to_pickle("data/embfeatures/feed_emb_vec_512.pickle")

    print("TruncatedSVD...")
    svd = TruncatedSVD(n_components=dim_n, n_iter=5, random_state=2009)
    svd.fit(emb)
    emb_svd = svd.transform(emb)
    df_emb = pd.DataFrame(emb_svd, columns=['512_femb'+str(i) for i in range(dim_n)])
    df_emb = pd.concat([feed_embeddings[['feedid']], df_emb], axis=1)
    print("to csv...")
    df_emb.to_pickle("data/embfeatures/feed_emb_vec_{dim}.pickle".format(dim=str(dim_n)))
else:
    print("data/embfeatures/feed_emb_vec_512.pickle")