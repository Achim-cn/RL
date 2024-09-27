import pandas as pd



# pd.options.display.max_columns = 100

total_action_df = pd.read_csv('../data/user_action.csv')


test_df = pd.read_csv('../data/test_a.csv')


print(test_df.shape)


# test_df['date_'] = total_action_df.date_.max() + 1


feed_info = pd.read_csv('../data/feed_info.csv')
feed_emb = pd.read_csv('../data/feed_embeddings.csv')

feed_info = feed_info[['feedid', 'videoplayseconds', 'authorid', 'bgm_song_id', 'bgm_singer_id']]

total_action_df = total_action_df.merge(feed_info, how='left', on=['feedid'])
test_df = test_df.merge(feed_info, how='left', on=['feedid'])

total_action_df['instance_id'] = range(len(total_action_df))
print(total_action_df)


test_df['instance_id'] = range(len(test_df))

test_df['instance_id'] = test_df['instance_id'] + total_action_df['instance_id'].max() + 1


total_action_df.to_pickle('../data/train.pkl')
test_df.to_pickle('../data/test_df.pkl')


