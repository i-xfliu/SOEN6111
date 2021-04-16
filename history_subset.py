from lastfm_dataset_1K_preprocess import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
import pyspark.sql.functions as sf
from pyspark.sql.functions import col

#most pupular songs history subset

#  "lastfm_dataset/top_20_percent_song_history.parquet"
#   extract the subset of listening history
#   top 20% popular songs listening history
#   history num : 7255562
#   track num: 33556
#   user num: 990

def init_spark(self):
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark

spark = init_spark(object)

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

def load_most_popular_song_listen_history():
    #20% songs
    history_df = load_lastfm_1k().load_useful_history()
    song_groupby = history_df.groupby('track_id', 'track_name').count().orderBy(col("count").desc())
    songs_top_20_percents = song_groupby.limit(round(song_groupby.count()*0.2))

    history_df = history_df.join(songs_top_20_percents.select('track_id'),['track_id'])
    print("history_df: %d" % history_df.count())
    history_df = history_df.select('index', 'user_id', 'timestamp', 'artist_id', 'artist_name', 'track_id', 'track_name')
    history_df.show()
    # history_df.write.parquet("lastfm_dataset/top_20_percent_song_history.parquet")
    return history_df



def load_most_active_users(k):
    history_df = load_lastfm_1k().load_useful_history()
    # 8297836
    # ['index', 'user_id', 'timestamp', 'artist_id', 'artist_name', 'track_id', 'track_name']
    #group by track_id
    user_groupby = history_df.groupby('user_id').count().orderBy(col("count").desc())
    user_groupby.show()
    # 'track_id','track_name','count'
    user_groupby = user_groupby.limit(k)
    return user_groupby

def load_users_placy_count():
    history_df = load_lastfm_1k().load_useful_history()
    # 8297836
    # ['index', 'user_id', 'timestamp', 'artist_id', 'artist_name', 'track_id', 'track_name']
    #group by track_id
    user_groupby = history_df.groupby('user_id').count().orderBy(col("count").desc())
    user_groupby.show()
    # 'track_id','track_name','count'
    return user_groupby

def load_tracks_placy_count():
    history_df = load_lastfm_1k().load_useful_history()
    # 8297836
    # ['index', 'user_id', 'timestamp', 'artist_id', 'artist_name', 'track_id', 'track_name']
    #group by track_id
    track_groupby = history_df.groupby('track_id').count().orderBy(col("count").desc())
    track_groupby.show()
    # 'track_id','track_name','count'
    return track_groupby

def load_top_20_song_history():

    history_df = spark.read.parquet("lastfm_dataset/top_20_percent_song_history.parquet")
    history_df.show()
    print("history num : %d"%history_df.count())
    print("track num: %d"%history_df.select('track_id').distinct().count())
    print("user num: %d"%history_df.select('user_id').distinct().count())
    # history num : 7255562
    # track num: 33556
    # user num: 990
    # +-------+-----------+--------------------+--------------------+-------------+--------------------+------------+
    # |  index|    user_id|           timestamp|           artist_id|  artist_name|            track_id|  track_name|
    # +-------+-----------+--------------------+--------------------+-------------+--------------------+------------+
    # |1834219|user_000089|2006-11-08 18:51:...|000fc734-b7e1-4a0...|Cocteau Twins|a95098a9-d0a9-434...|Aikea-Guinea|
    # |2667533|user_000135|2007-02-16 10:51:...|000fc734-b7e1-4a0...|Cocteau Twins|a95098a9-d0a9-434...|Aikea-Guinea|
    # |2665841|user_000135|2007-03-23 13:27:...|000fc734-b7e1-4a0...|Cocteau Twins|a95098a9-d0a9-434...|Aikea-Guinea|
    # |3260229|user_000159|2007-01-18 19:28:...|000fc734-b7e1-4a0...|Cocteau Twins|a95098a9-d0a9-434...|Aikea-Guinea|


    history_df = spark.read.parquet("lastfm_dataset/top_20_Percent_song_fractional_intID_history.parquet")
    history_df.show()
    print("user-track pair num： : %d"%history_df.count())
    print("track num: %d"%history_df.select('track_id').distinct().count())
    print("user num: %d"%history_df.select('user_id').distinct().count())
    # user-track pair num : 861902
    # track num: 33556
    # user num: 990
    # +-----------+--------------------+-------+--------+-----+--------------------+
    # |    user_id|            track_id|id_user|id_track|count|    fractional_count|
    # +-----------+--------------------+-------+--------+-----+--------------------+
    # |user_000577|008bf805-efd3-448...|      0|   10657|    1|2.065987645393880...|
    # |user_000706|008bf805-efd3-448...|     14|   10657|    1|1.234415504258733...|
    # |user_000112|008bf805-efd3-448...|     27|   10657|    1|1.731002250302925...|
    # |user_000670|008bf805-efd3-448...|     94|   10657|    3|4.447079750963534E-4|



    history_df = spark.read.parquet("lastfm_dataset/top_2_Percent_song_history.parquet")
    print(history_df.select('track_id').distinct().count()) #3381
    print(history_df.select('user_id').distinct().count()) #984









# load_most_popular_songs_set(1000)

# load_most_active_users_history()


# load_top_20_song_history()