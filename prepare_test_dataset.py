from sklearn.model_selection import KFold
import numpy as np

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import pyspark.sql.functions as sf
from pyspark.sql.functions import col
from history_subset import *
from pyspark.ml.recommendation import ALS, ALSModel


def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark
spark = init_spark()

df = spark.read.parquet("lastfm_dataset/top_20_Percent_song_fractional_intID_history.parquet")
df.show()
print(type(df))
(training, test) = df.randomSplit([0.8, 0.2],seed=100)
print(type(test))
test = test.limit(5000)

test_groupby = test.groupby('id_user').agg(count('*').alias('listened_song')).where(col('listened_song') > 30)
test_groupby = test_groupby.select(col("id_user"))
test = test.join(test_groupby, ['id_user'])
# test.write.parquet("lastfm_dataset/test_5000.parquet")

print("test user num: %d" % test.select("id_user").distinct().count())
print("make recomendation for ecah user from how many songs :%d" % test.select("id_track").distinct().count())

test_full = (
    test.select('id_user').distinct()
        .crossJoin(test.select('id_track').distinct())
        .join(test, on=['id_user', 'id_track'], how='left')
        .fillna(0, subset=['fractional_count','count'])
        .cache()
)

listend_song = test_full.where(col('fractional_count') > 0).groupby('id_user').agg(count('*').alias('listened_song'))
nolistend_song = test_full.where(col('fractional_count') == 0).groupby('id_user').agg(count('*').alias('not_listened'))
df = listend_song.join(nolistend_song, ['id_user'])
print(df.count())
df.show()

# test_full.write.parquet("lastfm_dataset/test_full_0.01.parquet")

