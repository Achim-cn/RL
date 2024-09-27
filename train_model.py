# coding: utf-8

from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow.keras import initializers,regularizers,constraints
from tensorflow.keras import layers, Model
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.backend import expand_dims, repeat_elements, sum
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import *
# from tensorflow.compat.v1.keras.layers import *
import tensorflow.keras.backend as K

import pandas as pd
import numpy as np
from tqdm import tqdm

import warnings,os,gc
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
path = 'data'
if not os.path.exists(path + "/embfeatures"):
    os.mkdir(path + "/embfeatures")
if not os.path.exists(path + "/id2index"):
    os.mkdir(path + "/id2index")

#================================================= Data =================================================
feed_info = pd.read_csv(path + '/feed_info.csv')
user_action = pd.read_csv(path + '/user_action.csv').drop_duplicates(subset=["userid", "feedid"], keep="last").reset_index(drop=True)
label_cols = ['read_comment','like','click_avatar','forward','favorite','comment','follow']+['play']

#================================================= ID2Index =================================================
def id_encode(series):          # 工业课一般不用这一步
    unique = list(series.unique())
    unique.sort()
    return dict(zip(unique, range(series.nunique()))) #, dict(zip(range(series.nunique()), unique))

def user_graph(df):
    if not os.path.exists(path + "/embfeatures/user_graph.pickle"):
        tmp = df[df.follow==1][["userid","authorid"]].drop_duplicates()
        tmp['logic']=0.0
        tmp.to_pickle(path + "/embfeatures/user_graph.pickle")

if not os.path.exists(path + "/id2index/userid2index.npy"):
    userid2index = id_encode(user_action.userid)    # key: userid,  value: index
    i = 0
    # for k, v in userid2index.items():
    #     print(k, v)
    #     i += 1
    #     if i == 10:
    #         break
    # print('-------------------------')
    feedid2index = id_encode(feed_info.feedid)
    # print(feedid2index)
    # print('*****************************')
    authorid2index = id_encode(feed_info.authorid)
    # print(authorid2index)
    np.save(path + "/id2index/userid2index.npy",userid2index)
    np.save(path + "/id2index/feedid2index.npy",feedid2index)
    np.save(path + "/id2index/authorid2index.npy",authorid2index)
else:
    userid2index = np.load(path + "/id2index/userid2index.npy",allow_pickle=True).item()
    feedid2index = np.load(path + "/id2index/feedid2index.npy",allow_pickle=True).item()
    authorid2index = np.load(path + "/id2index/authorid2index.npy",allow_pickle=True).item()
    
def read_id_dict(name, dim_n, emb_mode):
    tmp = pd.read_pickle(path + '/embfeatures/{name}_{mode}_{dim}.pickle'.format(name=name,mode=emb_mode,dim=str(dim_n)))
    tmp_dict = {}
    for i,item in zip(tmp[tmp.columns[0]].values, tmp[tmp.columns[1:]].values):
        tmp_dict[i] = item
    return tmp_dict

def embedding_mat(feat_name, dim_n, emb_mode):
    model = read_id_dict(feat_name,dim_n,emb_mode)
    if feat_name.startswith("user"):
        id2index = userid2index
    elif feat_name.startswith("feed"):
        id2index = feedid2index
    elif feat_name.startswith("author"):
        id2index = authorid2index
    else:
        print("Feat Name Error!!!")
    
    embed_matrix = np.zeros((len(id2index) + 1, dim_n))
    for word, i in id2index.items():
        embedding_vector = model[word] if word in model else None
        if embedding_vector is not None:
            embed_matrix[i] = embedding_vector
        else:
            unk_vec = np.random.random(dim_n) * 0.5
            unk_vec = unk_vec - unk_vec.mean()
            embed_matrix[i] = unk_vec
    
    return embed_matrix

def pad_seq(df, feat_name, max_len=1, mode='data'):
    tmp = df[[feat_name]].copy()
    # print(tmp)
    # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    if feat_name.startswith("user"):
        id2index = userid2index
    elif feat_name.startswith("feed"):
        id2index = feedid2index
    elif feat_name.startswith("author"):
        id2index = authorid2index
    else:
        print("Feat Name Error!!!")

    tmp[feat_name] = tmp[feat_name].apply(lambda x:id2index[x])
    # print(tmp)
    # print('************************')
    seq = pad_sequences(tmp.values, maxlen = max_len)
    # print(seq.size)

    return seq

#================================================= Embedding Weight =================================================
dim_n = 150
emb_m = "vec"
user_feed_emb = embedding_mat("user_feed",dim_n,emb_m)
user_author_emb = embedding_mat("user_author",dim_n,emb_m)
feed_user_emb = embedding_mat("feed_user",dim_n,emb_m)
author_user_emb = embedding_mat("author_user",dim_n,emb_m)

dim_n = 128
emb_m = "vec"
user_key1_emb = embedding_mat("user_key1",dim_n,emb_m)
feed_key1_emb = embedding_mat("feed_key1",dim_n,emb_m)

user_key2_emb = embedding_mat("user_key2",dim_n,emb_m)
feed_key2_emb = embedding_mat("feed_key2",dim_n,emb_m)

user_tag_emb = embedding_mat("user_tag",32,emb_m)
feed_tag_emb = embedding_mat("feed_tag",32,emb_m)

feed_emb = embedding_mat("feed_emb",150,emb_m)

dim_n = 150
emb_m = "d2v"
user_feed_d2v = embedding_mat("user_feed",dim_n,emb_m)
user_author_d2v = embedding_mat("user_author",dim_n,emb_m)
feed_user_d2v = embedding_mat("feed_user",dim_n,emb_m)
author_user_d2v = embedding_mat("author_user",dim_n,emb_m)

#================================================= Dataset =================================================
data_set = user_action[["userid","feedid"]+label_cols].drop_duplicates(subset=["userid","feedid"],keep="last").reset_index(drop=True)
data_set = data_set.merge(feed_info[["feedid","authorid","videoplayseconds"]], how='left',on="feedid")
# print(data_set)

data_set["play"] = data_set["play"]/1000
data_set["play"] = data_set["play"]/data_set["videoplayseconds"]
data_set["play"] = data_set["play"].apply(lambda x:1 if x>0.9 else 0)
data_set = data_set.astype(int)
print(data_set)
print('*****************************')
# user_graph(data_set)

data_userid = pad_seq(data_set, "userid", mode="data")
# print((data_userid))
# print('**************************************')
data_feedid = pad_seq(data_set, "feedid", mode="data")
# print(data_feedid)
# print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
data_authorid = pad_seq(data_set, "authorid", mode="data")
# print(data_authorid)
# print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
data_feats = np.concatenate([data_userid,data_feedid,data_authorid],axis=1)
print(data_feats)
print('----------------data feats------------------------')
data_labels = data_set[label_cols].values.reshape(-1,8)
print(data_labels)
print('----------------data labels-----------------------')


#================================================= Model =================================================
# class GenTrainTestData(Sequence):
#     def __init__(self, batch_indexs, batch_size=1024, shuffle=True, data_x=[], data_y=[]):
#         'Initialization'
#         self.batch_size   = batch_size
#         self.batch_indexs = batch_indexs
#         self.shuffle      = shuffle
#         self.data_x       = data_x
#         self.data_y       = data_y
#         self.on_epoch_end()
#
#     def __len__(self): # The number of batches per epoch
#         len_ = int(np.floor(len(self.batch_indexs)/self.batch_size))
#         return len_ if len(self.batch_indexs) % self.batch_size == 0 else len_+1
#
#     def __getitem__(self, index):
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
#         batch_indexs_tmp = [self.batch_indexs[k] for k in indexes]
#         X, y = self.__data_generation(batch_indexs_tmp)
#         return X, y
#
#     def on_epoch_end(self): # Update
#         self.indexes = np.arange(len(self.batch_indexs))
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)
#
#     def __data_generation(self, batch_indexs_tmp):
#         X_ = self.data_x[batch_indexs_tmp]
#         y_ = self.data_y[batch_indexs_tmp]
#         return X_, y_
    
class MultiTask(tf.keras.Model):
    def __init__(self, units, num_experts, num_tasks, 
                 use_expert_bias=True,use_gate_bias=True,expert_activation='relu', gate_activation='softmax',
                 expert_bias_initializer='zeros',gate_bias_initializer='zeros',expert_bias_regularizer=None, 
                 gate_bias_regularizer=None, expert_bias_constraint=None,gate_bias_constraint=None,
                 expert_kernel_initializer='VarianceScaling', gate_kernel_initializer='VarianceScaling',
                 expert_kernel_regularizer=None,gate_kernel_regularizer=None,expert_kernel_constraint=None,
                 gate_kernel_constraint=None,activity_regularizer=None, **kwargs):
        super(MultiTask, self).__init__(**kwargs)
        self.units = units
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.expert_kernels = None
        self.gate_kernels = None
        self.expert_kernel_initializer = initializers.get(expert_kernel_initializer)
        self.gate_kernel_initializer = initializers.get(gate_kernel_initializer)
        self.expert_kernel_regularizer = regularizers.get(expert_kernel_regularizer)
        self.gate_kernel_regularizer = regularizers.get(gate_kernel_regularizer)
        self.expert_kernel_constraint = constraints.get(expert_kernel_constraint)
        self.gate_kernel_constraint = constraints.get(gate_kernel_constraint)
        self.expert_activation = expert_activation
        self.gate_activation = gate_activation
        self.expert_bias = None
        self.gate_bias = None
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.expert_bias_initializer = initializers.get(expert_bias_initializer)
        self.gate_bias_initializer = initializers.get(gate_bias_initializer)
        self.expert_bias_regularizer = regularizers.get(expert_bias_regularizer)
        self.gate_bias_regularizer = regularizers.get(gate_bias_regularizer)
        self.expert_bias_constraint = constraints.get(expert_bias_constraint)
        self.gate_bias_constraint = constraints.get(gate_bias_constraint)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.expert_layers = []
        self.gate_layers = []
        
        for i in range(self.num_experts):
            self.expert_layers.append(layers.Dense(self.units, 
                                                   activation=self.expert_activation,
                                                   use_bias=self.use_expert_bias,
                                                   kernel_initializer=self.expert_kernel_initializer,
                                                   bias_initializer=self.expert_bias_initializer,
                                                   kernel_regularizer=self.expert_kernel_regularizer,
                                                   bias_regularizer=self.expert_bias_regularizer,
                                                   activity_regularizer=None,
                                                   kernel_constraint=self.expert_kernel_constraint,
                                                   bias_constraint=self.expert_bias_constraint))
        for i in range(self.num_tasks):
            self.gate_layers.append(layers.Dense(self.num_experts, 
                                                 activation=self.gate_activation,
                                                 use_bias=self.use_gate_bias,
                                                 kernel_initializer=self.gate_kernel_initializer,
                                                 bias_initializer=self.gate_bias_initializer,
                                                 kernel_regularizer=self.gate_kernel_regularizer,
                                                 bias_regularizer=self.gate_bias_regularizer, 
                                                 activity_regularizer=None,
                                                 kernel_constraint=self.gate_kernel_constraint,
                                                 bias_constraint=self.gate_bias_constraint))
    def call(self, inputs):
        expert_outputs, gate_outputs, final_outputs = [], [], []
        for expert_layer in self.expert_layers:
            expert_output = tf.expand_dims(expert_layer(inputs), axis=2)
            expert_outputs.append(expert_output)
        expert_outputs = tf.concat(expert_outputs,2)

        for gate_layer in self.gate_layers:
            gate_outputs.append(gate_layer(inputs))
            
        for gate_output in gate_outputs:
            expanded_gate_output = tf.expand_dims(gate_output, axis=1)
            weighted_expert_output = expert_outputs * repeat_elements(expanded_gate_output, self.units, axis=1)
            final_outputs.append(sum(weighted_expert_output, axis=2))
        return final_outputs    #dmt  (1,2,3), (4,5,6),(7,8,9)
                                #     (0.5, 0.2, 0.3)  (0.4, 0.3, 0.3)
                                #  dmt[0] = 0.5 * (1,2,3) + 0.2 * (4,5,6) + 0.3 * (7,8,9)
                                #  dmt[1] = 0.4 * (1,2,3) + 0.3 * (4,5,6) + 0.3 * (7,8,9)
    
class DotaModelM1V128K10(Model):
    def __init__(self,):
        super(DotaModelM1V128K10, self).__init__()
        # Class Embedding Layers
        self.UserFeedEmbedding = Embedding(input_dim=user_feed_emb.shape[0], output_dim=user_feed_emb.shape[1],embeddings_initializer = tf.keras.initializers.constant(user_feed_emb),dtype='float32')
        self.UserAuthorEmbedding = Embedding(input_dim=user_author_emb.shape[0], output_dim=user_author_emb.shape[1],embeddings_initializer = tf.keras.initializers.constant(user_author_emb),dtype='float32')
        self.UserTagEmbedding = Embedding(input_dim=user_tag_emb.shape[0], output_dim=user_tag_emb.shape[1],embeddings_initializer = tf.keras.initializers.constant(user_tag_emb),dtype='float32')
        self.UserKey1Embedding = Embedding(input_dim=user_key1_emb.shape[0], output_dim=user_key1_emb.shape[1],embeddings_initializer = tf.keras.initializers.constant(user_key1_emb),dtype='float32')
        self.UserKey2Embedding = Embedding(input_dim=user_key2_emb.shape[0], output_dim=user_key2_emb.shape[1],embeddings_initializer = tf.keras.initializers.constant(user_key2_emb),dtype='float32')
        self.FeedUserEmbedding = Embedding(input_dim=feed_user_emb.shape[0], output_dim=feed_user_emb.shape[1],embeddings_initializer = tf.keras.initializers.constant(feed_user_emb), dtype='float32')
        self.FeedTagEmbedding = Embedding(input_dim=feed_tag_emb.shape[0], output_dim=feed_tag_emb.shape[1],embeddings_initializer = tf.keras.initializers.constant(feed_tag_emb), dtype='float32')
        self.FeedKey1Embedding = Embedding(input_dim=feed_key1_emb.shape[0], output_dim=feed_key1_emb.shape[1],embeddings_initializer = tf.keras.initializers.constant(feed_key1_emb), dtype='float32')
        self.FeedKey2Embedding = Embedding(input_dim=feed_key2_emb.shape[0], output_dim=feed_key2_emb.shape[1],embeddings_initializer = tf.keras.initializers.constant(feed_key2_emb), dtype='float32')
        self.FeedEmbEmbedding = Embedding(input_dim=feed_emb.shape[0], output_dim=feed_emb.shape[1],embeddings_initializer = tf.keras.initializers.constant(feed_emb),trainable=False,dtype='float32')
        self.AuthorUserEmbedding = Embedding(input_dim=author_user_emb.shape[0], output_dim=author_user_emb.shape[1],embeddings_initializer = tf.keras.initializers.constant(author_user_emb),dtype='float32')
        # Class D2vEmbedding Layers
        self.UserFeedD2vEmbedding = Embedding(input_dim=user_feed_d2v.shape[0], output_dim=user_feed_d2v.shape[1],embeddings_initializer = tf.keras.initializers.constant(user_feed_d2v),dtype='float32')
        self.UserAuthorD2vEmbedding = Embedding(input_dim=user_author_d2v.shape[0], output_dim=user_author_d2v.shape[1],embeddings_initializer = tf.keras.initializers.constant(user_author_d2v),dtype='float32')
        self.FeedUserD2vEmbedding = Embedding(input_dim=feed_user_d2v.shape[0], output_dim=feed_user_d2v.shape[1],embeddings_initializer = tf.keras.initializers.constant(feed_user_d2v), dtype='float32')
        self.AuthorUserD2vEmbedding = Embedding(input_dim=author_user_d2v.shape[0], output_dim=author_user_d2v.shape[1],embeddings_initializer = tf.keras.initializers.constant(author_user_d2v),dtype='float32')
        # Class Base Layers
        self.DotaDense256 = Dense(256, activation='relu')
        self.DotaDense128 = Dense(128, activation='relu')
        self.DotaDropout3 = Dropout(0.3)
        self.DotaDropout2 = Dropout(0.2)
        # Class MultiTaskLayer
        self.DotaMTLayer = MultiTask(units=128, num_experts=4, num_tasks=8)
        self.DotaMTTower1s = []
        self.DotaMTDropouts1s = []
        self.DotaMTTower2s = []
        self.DotaMTDenseSigs = []
        # ['read_comment','like','click_avatar','forward','favorite','comment','follow']
        for i in range(8):
            self.DotaMTTower1s.append(Dense(128, activation='relu'))
            self.DotaMTDropouts1s.append(Dropout(0.2))
            self.DotaMTTower2s.append(Dense(128, activation='relu'))
            self.DotaMTDenseSigs.append(Dense(1, activation='sigmoid'))
        
    def call(self, inputs):
        inputs = tf.reshape(inputs, shape = [-1, 3])
        user_feed_embed = self.UserFeedEmbedding(inputs[:,0])
        user_author_embed = self.UserAuthorEmbedding(inputs[:,0])
        user_tag_embed = self.UserTagEmbedding(inputs[:,0])
        user_key1_embed = self.UserKey1Embedding(inputs[:,0])
        user_key2_embed = self.UserKey2Embedding(inputs[:,0])
        feed_user_embed = self.FeedUserEmbedding(inputs[:,1])
        feed_tag_embed = self.FeedTagEmbedding(inputs[:,1])
        feed_key1_embed = self.FeedKey1Embedding(inputs[:,1])
        feed_key2_embed = self.FeedKey2Embedding(inputs[:,1])
        feed_emb_embed = self.FeedEmbEmbedding(inputs[:,1])
        author_user_embed = self.AuthorUserEmbedding(inputs[:,2])
        
        user_feed_d2vem = self.UserFeedD2vEmbedding(inputs[:,0])
        user_author_d2vem = self.UserAuthorD2vEmbedding(inputs[:,0])
        feed_user_d2vem = self.FeedUserD2vEmbedding(inputs[:,1])
        author_user_d2vem = self.AuthorUserD2vEmbedding(inputs[:,2])
        
        user_w_feed = tf.matmul(tf.expand_dims(user_feed_d2vem,-1), tf.expand_dims(feed_user_d2vem,-1), transpose_b = True)
        user_w_auth = tf.matmul(tf.expand_dims(user_author_d2vem,-1), tf.expand_dims(author_user_d2vem,-1), transpose_b = True)
        user_w_user = tf.concat([user_w_feed,user_w_auth],axis=1)
        user_w_user = tf.reshape(user_w_user, shape = [-1,user_w_user.shape[1] * user_w_user.shape[2]])
        user_w_user = self.DotaDropout3(self.DotaDense128(user_w_user))

        user_x_feed = tf.matmul(tf.expand_dims(user_feed_embed,-1), tf.expand_dims(feed_user_embed,-1), transpose_b = True)
        user_x_auth = tf.matmul(tf.expand_dims(user_author_embed,-1), tf.expand_dims(author_user_embed,-1), transpose_b = True)
        user_x_user = tf.concat([user_x_feed,user_x_auth],axis=1)
        user_x_user = tf.reshape(user_x_user, shape = [-1, user_x_user.shape[1] * user_x_user.shape[2]])
        user_x_user = self.DotaDropout2(self.DotaDense256(user_x_user))
        
        dmt = self.DotaMTLayer(tf.concat([user_feed_embed,user_author_embed,user_tag_embed,user_key1_embed,user_key2_embed,user_feed_d2vem,user_author_d2vem,
                                          feed_user_embed,feed_tag_embed,feed_key1_embed,feed_emb_embed,feed_key2_embed,feed_user_d2vem,
                                          author_user_embed,author_user_d2vem,
                                          user_x_user,user_w_user],axis=-1))

        outputs = []
        print('-----------dmt---------------')
        print(dmt)
        for task, action in enumerate(dmt):                    #不同任务网络
            x =  self.DotaMTDropouts1s[task](self.DotaMTTower1s[task](action))
            x =  self.DotaMTTower2s[task](x) 
            x =  self.DotaMTDenseSigs[task](x)
            outputs.append(x)

        return outputs # tf.concat(outputs,axis=-1)
     
    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        _ = self.call(inputs)

#================================================= Training Argv =================================================
name = "m1v128k10"
phase_feat = "semi"
epochs_ = 1
kfolds_ = 10
lr_rate = 1e-3
batchsize_ = 1024 * 20

if not os.path.exists(path + "/model"):
    os.mkdir(path + "/model")

if not os.path.exists(path + "/model/" + name):
    os.mkdir(path + "/model/" + name)

#================================================= Training =================================================
kf = KFold(n_splits = kfolds_, shuffle=True, random_state=2009).split(data_feats)

for i, (train_fold, valid_fold) in enumerate(kf): 
    print(i, 'fold')
    train_feats = data_feats[train_fold]
    train_labels = data_labels[train_fold]
    print(train_feats.shape)
    #=========================================== fit =================================================
    the_path = path + "/model/{name}/{flag}_k{n}".format(name=name, flag=phase_feat, n=str(i))
    if not os.path.exists(the_path):
        os.mkdir(the_path)
    
    model = DotaModelM1V128K10()
    checkpoint = ModelCheckpoint(the_path + '/model.h5', monitor='loss', verbose=1, save_best_only=True, mode='min',save_weights_only=False)
    csv_logger = CSVLogger(the_path + '/log_{f}.log'.format(f=str(i)))
    # model.compile(optimizer=tf.optimizers.Nadam(lr=lr_rate),
    #           loss=[tf.keras.losses.binary_crossentropy for i in range(8)],
    #           loss_weights=[1, 1, 1, 1, 1, 1, 1, 0.1])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_rate),
                  loss=[tf.keras.losses.binary_crossentropy for i in range(8)],
                  loss_weights=[1, 1, 1, 1, 1, 1, 1, 0.1])
    # print(train_feats)
    model.fit(train_feats, 
              [train_labels[:,i] for i in range(8)], 
              batch_size=batchsize_, 
              epochs = epochs_,
              callbacks=[checkpoint,csv_logger])
    # model.save(path+'/model/')


    # del model
    # gc.collect()
    # K.clear_session()
    