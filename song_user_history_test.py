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
from lastfm_dataset_1K_preprocess import load_lastfm_1k
from lastfm_subset_jason_preprocess import load_lastfm_subset
import sqlite3
from pyspark import SparkContext
sc = SparkContext.getOrCreate()


def init_spark(self):
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark

spark = init_spark(object)



db_path = '/Users/xiaoyunliao/Documents/study/concordia/soen6111/project/database/track_metadata.db'

# query = 'SELECT * from songs'
# conn = sqlite3.connect(db_path)
#
# a_pandas_df = pd.read_sql_query(query, conn)
# print(type(a_pandas_df))
# print(a_pandas_df.columns)
# songs_df = spark.createDataFrame(a_pandas_df)
# print(songs_df.columns)
# print(songs_df.count())
# print(songs_df.head())
#
#
# history_df = load_lastfm_1k().load_user_history()
# print(history_df.columns)
# song_df = history_df.select('track_id','artist_id','artist_name','track_name').distinct()
# print(song_df.columns)
# print(song_df.count())
#
# #比较history和 lastfm
#
# intersection_df = songs_df.join(history_df, (songs_df.artist_mbid == history_df.artist_id) & (songs_df.title == history_df.track_name), how="right")
#
#
# print(intersection_df.filter(intersection_df['artist_mbid'] is None).count())

def get_unique_artist_track_history():
    db_path = '/Users/xiaoyunliao/Documents/study/concordia/soen6111/project/database/track_metadata.db'
    query = 'SELECT distinct artist_id,artist_name,track_id,track_name from history'
    conn = sqlite3.connect(db_path)
    a_pandas_df = pd.read_sql_query(query, conn)
    a_pandas_df.to_parquet("unique_song_history.parquet")

get_unique_artist_track_history()

from sqlalchemy import create_engine
def write_history_db():

    engine = create_engine('sqlite:////Users/xiaoyunliao/Documents/study/concordia/soen6111/project/database/track_metadata.db', echo=True)
    sqlite_connection = engine.connect()

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
    sqlite_table = "history"
    df_user_track.to_sql(sqlite_table, sqlite_connection, if_exists='fail')



