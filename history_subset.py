from lastfm_dataset_1K_preprocess import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
import pyspark.sql.functions as sf
from pyspark.sql.functions import col

#most pupular songs history subset

# 1.select most 5000 popular songs
# 2. according to the popular songs set, select users which have enough listening history

def load_most_popular_songs_set(k):
    # top k songs
    history_df = load_lastfm_1k().load_useful_history()
    # 8297836
    # ['index', 'user_id', 'timestamp', 'artist_id', 'artist_name', 'track_id', 'track_name']
    #group by track_id
    song_groupby = history_df.groupby('track_id','track_name').count().orderBy(col("count").desc())
    song_groupby.show()
    # 'track_id','track_name','count'
    song_groupby = song_groupby.limit(k)
    return song_groupby

def load_most_active_users_history():
    #20% songs
    history_df = load_lastfm_1k().load_useful_history()
    song_groupby = history_df.groupby('track_id', 'track_name').count().orderBy(col("count").desc())
    songs_top_20_percents = song_groupby.limit(round(song_groupby.count()*0.2))
    print("songs_top_20_percents: %d"%songs_top_20_percents.count())
    users_listen_top_song= history_df.join(songs_top_20_percents,['track_id']).groupby('user_id').count().orderBy(col("count").desc())
    users_listen_top_song.show()
    users_top_20_percents = users_listen_top_song.limit(round(users_listen_top_song.count()*0.2))
    print("users_top_20_percents: %d" % users_top_20_percents.count())
    users_top_20_percents.show()
    history_df = history_df.join(songs_top_20_percents.select('track_id'),['track_id'])
    history_df = history_df.join(users_top_20_percents.select('user_id'),['user_id'])
    print("history_df: %d" % history_df.count())
    history_df = history_df.select('index', 'user_id', 'timestamp', 'artist_id', 'artist_name', 'track_id', 'track_name')
    history_df.show()
    return history_df









# load_most_popular_songs_set(1000)

# load_most_active_users_history()