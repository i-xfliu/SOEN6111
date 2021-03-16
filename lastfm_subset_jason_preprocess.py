import os
# Spark imports
from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
# Dask imports
# import dask.bag as db
# import dask.dataframe as df_user_track  # you can use Dask bags or dataframes
from csv import reader
import pandas as pd
import re
import pyarrow._parquet as _parquet


# http://millionsongdataset.com/lastfm/
# 10k tracks, each one has multiple similar tracks and tags
# ['artist', 'similars', 'tags', 'timestamp', 'title', 'track_id']

class load_lastfm_subset(object):

    def init_spark(self):
        spark = SparkSession \
            .builder \
            .appName("Python Spark SQL basic example") \
            .config("spark.some.config.option", "some-value") \
            .getOrCreate()
        return spark

    spark = init_spark(object)


    class load_all_json_file(object):
        file_path_list = []
        def get_file_path(self, dir_parth):
            file_list = [f'{dir_parth}/{x.strip()}' for x in os.listdir(dir_parth)]
            for path in file_list:
                if path[-4:] == 'json':
                    self.file_path_list.append(path)
                elif path[-5:] == 'Store':
                    print()
                else:
                    self.get_file_path(path)


    # cover json to parquet
    def read_json_to_parquet(self):
        loader = self.load_all_json_file()
        # json folder has deleted
        loader.get_file_path("data/lastfm-sub_data/lastfm_subset")
        all_file_path = loader.file_path_list
        lastfm_df = self.spark.read.json(all_file_path)
        lastfm_df.write.option("parquet.block.size", 10 * 1024 * 1024)
        lastfm_df.write.parquet("data/track_artist_sml_tag")


    def load_lastfm_subset_parquet(self):
        # load data from parquet fold
        lastfm_df = self.spark.read.parquet("data/track_artist_sml_tag/*.parquet")

        print(lastfm_df.count())
        # 9330
        print(lastfm_df.columns)
        # ['artist', 'similars', 'tags', 'timestamp', 'title', 'track_id']
        print(lastfm_df.head())
        # artist='The Shirelles',
        # similars=[
        # ['TRCCSCE128F92EF9A9', '1'],
        # ['TRCERNU128F92E1921', '1'],
        # ['TRFGOKM128F92F13FF', '1'],
        # ['TRFXYRO128F9311F1F', '1'],
        # ['TRQIIYJ128F92FC318', '1'],
        # ['TRYZQVK128F92FE86E', '0.87287'] ...]
        # tags=[['oldies', '100'], ['60s', '90'] ....],
        # timestamp='2011-09-07 13:30:37.182021',
        # title='Dedicated To the One I Love',
        # track_id='TRAHBWE128F9349247'

        print(lastfm_df.schema)
        # StructType(
        # List(StructField(artist,StringType,true),
        # StructField(similars,ArrayType(ArrayType(StringType,true),true),true),
        # StructField(tags,ArrayType(ArrayType(StringType,true),true),true),
        # StructField(timestamp,StringType,true),
        # StructField(title,StringType,true),
        # StructField(track_id,StringType,true))
        # )

        return lastfm_df

# load_lastfm_subset().load_lastfm_subset_parquet()