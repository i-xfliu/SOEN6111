# Spark imports
from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
# Dask imports
import dask.bag as db
import dask.dataframe  # you can use Dask bags or dataframes
from csv import reader
import pandas as pd
import re
import pyarrow._parquet as _parquet
import sqlite3

# userid-profile.tsv
# lastfm-dataset-1k.snappy.parquet
# http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html
# user - 992 rows, ['#id', 'gender', 'age', 'country', 'registered']
# history -19098862 rows, ['user_id', 'timestamp', 'artist_id', 'artist_name', 'track_id', 'track_name']

#Initialize a spark session.


class load_lastfm_1k(object):

    def init_spark(self):
        spark = SparkSession \
            .builder \
            .appName("Python Spark SQL basic example") \
            .config("spark.some.config.option", "some-value") \
            .getOrCreate()
        return spark

    spark = init_spark(object)
    db_path = '/Users/xiaoyunliao/Downloads/track_metadata.db'
    conn = sqlite3.connect(db_path)


    # return spark df
    def load_userid_profile(self):
        df_user_profile = self.spark.read.csv('data/lastfm-dataset-1k/userid-profile.tsv',sep='\t', header=True)
        print(df_user_profile.count())
        # 992
        print(df_user_profile.columns)
        # ['#id', 'gender', 'age', 'country', 'registered']
        print(df_user_profile.head())
        # Row(#id='user_000001', gender='m', age=None, country='Japan', registered='Aug 13, 2006')
        return df_user_profile


    def  load_user_history(self):
        # large single file
        filepath = "data/lastfm-dataset-1k/lastfm-dataset-1k.snappy.parquet"
        df_user_track = self.spark.read.parquet(filepath)
        df_user_track.show()
        # print(df_user_track.count())
        # 19098862
        # ['user_id', 'timestamp', 'artist_id', 'artist_name', 'track_id', 'track_name']
        # print(df_user_track.columns)
        # Row(
        # user_id='user_000001',
        # timestamp=datetime.datetime(2006, 8, 13, 9, 59, 20),
        # artist_id='09a114d9-7723-4e14-b524-379697f6d2b5',
        # artist_name='Plaid & Bob Jaroc',
        # track_id='c4633ab1-e715-477f-8685-afa5f2058e42',
        # track_name='The Launching Of Big Face')
        # print(df_user_track.head())
        # print(df_user_track.schema)
        # StructType(
        # List(StructField(user_id,StringType,true),
        # StructField(timestamp,TimestampType,true),
        # StructField(artist_id,StringType,true),
        # StructField(artist_name,StringType,true),
        # StructField(track_id,StringType,true),
        # StructField(track_name,StringType,true)))

        # user_track_df = df_user_track.groupby('user_id', 'track_id').count()
        # Row(
        # user_id='user_000001',
        # track_id='7b793966-abc8-423d-bddc-1a761213d5f7',
        # count=10)
        # print(user_track_df.head())
        # df_user_track.select('artist_id', 'artist_name', 'track_id', 'track_name').distinct().write.parquet("history_song_set_parquet")

        return df_user_track

    def load_history_song_set_parquet(self):
        filepath = "lastfm_dataset/unique_song_history.parquet"
        history_song_set_df = self.spark.read.parquet(filepath)
        print(history_song_set_df.count())
        print(history_song_set_df.head())
        # 1503593
        # Row(artist_id='0003fd17-b083-41fe-83a9-d550bd4f00a1', artist_name='安倍なつみ',
        #     track_id='e85b51d3-895c-4fca-98c0-467ce557b48a', track_name='恋のテレフォンGoal')
        return history_song_set_df

    def extract_common_song_info(self):
        history_df = self.load_history_song_set_parquet()
        query = 'SELECT * from songs'
        songs_df = pd.read_sql_query(query, self.conn)
        songs_df = self.spark.createDataFrame(songs_df)
        intersection_df = songs_df.join(history_df, (songs_df.artist_mbid == history_df.artist_id) & (
                    songs_df.title == history_df.track_name), how="left_semi")
        print(intersection_df.count())
        df = intersection_df.toPandas()
        df.to_sql('common_song', self.conn, if_exists='fail')
        # intersection_df.write.parquet("common_song_dataset_parquet")

    def write_lastfm_common_songs_to_parquet(self):
        query = 'SELECT * from common_song'
        songs_df = pd.read_sql_query(query, self.conn)
        songs_df.to_parquet("lastfm_common_song_dataset.parquet")

    def extract_useful_history_to_parquet(self):
        # 'artist_id', 'artist_name', 'track_id', 'track_name'
        # songs_df.artist_mbid == history_df.artist_id
        # songs_df.title == history_df.track_name
        query = "select h.* from history h, common_song c  where h.artist_id = c.artist_mbid and h.track_name = c.title"
        history_df = pd.read_sql_query(query, self.conn)
        history_df.to_parquet("useful_history_1k.parquet")

    def write_history_to_parquet(self):
        # loading tsv by pandas, then cover to parquet

        df_user_track = pd.read_csv(
            "data/userid-timestamp-artid-artname-traid-traname.tsv", sep='\t', header=None,
            names=[
                'user_id', 'timestamp', 'artist_id', 'artist_name', 'track_id', 'track_name'
            ],
            # skiprows=[
            #     2120260-1, 2446318-1, 11141081-1,
            #     11152099-1, 11152402-1, 11882087-1,
            #     12902539-1, 12935044-1, 17589539-1
            # ]
            error_bad_lines=False
        )
        df_user_track["timestamp"] = pd.to_datetime(df_user_track.timestamp)
        df_user_track.sort_values(['user_id', 'timestamp'], ascending=True, inplace=True)
        print(
            f'Number of Records: {len(df_user_track):,}\nUnique Users: {df_user_track.user_id.nunique()}\nUnique Artist:{df_user_track.artist_id.nunique():,}')

        save_filepath = "data/lastfm-dataset-1k_parquet"
        sd = dask.dataframe.from_pandas(df_user_track, npartitions=120)
        sd.to_parquet(save_filepath, compression={"name": "gzip", "values": "snappy"})
        df_user_track.to_parquet(save_filepath, compression='snappy', index=False)


    def load_useful_history(self):
        path = 'lastfm_dataset/useful_history_1k.parquet'
        df = self.spark.read.parquet(path)
        print("load_useful_history")
        # 8297836
        # ['index', 'user_id', 'timestamp', 'artist_id', 'artist_name', 'track_id', 'track_name']
        print("Rows:%d"%df.count())
        # print(df.columns)
        df.show()
        return df


    def load_lastfm_common_songs_parquet(self):
        path = 'lastfm_dataset/lastfm_common_song_dataset.parquet'
        df = self.spark.read.parquet(path)
        print("load_lastfm_common_songs")
        # 197549
        # ['index', 'track_id', 'title', 'song_id', 'release', 'artist_id',
        # 'artist_mbid', 'artist_name', 'duration',
        #  'artist_familiarity', 'artist_hotttnesss', 'year', 'track_7digitalid', 'shs_perf', 'shs_work']
        print("Rows:"+df.count())
        print("columns:"+df.columns)
        df.show()
        return df





# load_lastfm_1k().load_history_song_set_parquet()
# load_lastfm_1k().extract_common_song_info()
# load_lastfm_1k().write_lastfm_common_songs_to_parquet()
# load_lastfm_1k().extract_useful_history_to_parquet()
# load_lastfm_1k().load_user_history()

# load_lastfm_1k().load_userid_profile()
# load_lastfm_1k().load_useful_history()
# load_lastfm_1k().load_lastfm_common_songs_parquet()





