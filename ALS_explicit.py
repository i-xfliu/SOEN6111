from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import pyspark.sql.functions as sf
from pyspark.sql.functions import col
from history_subset import *
from pyspark.ml.recommendation import ALS, ALSModel

# "implicit feedback" problem
# how to transfer count to rating
# option 1: fractional_count

# 1. loading dataset - top_20_percent_song_history.parquet, including 990 users and 33556 tracks, 7255562 history
# 2. adding int user id and track id （ALS只接受数值型id）
# 3. adding fractional_count （用fractional_count 来作为预测值）
# 4. saving df to "lastfm_dataset/top_20_Percent_song_fractional_intID_history.parquet" (之后测试可以直接load这个df，节省时间)
# 5. train ALS model
# 6. parameter : userCol="id_user", itemCol="id_track", ratingCol="fractional_count", implicitPrefs=False
# 7. evaluation:  using Mean Percentage Ranking ( 𝑀𝑃𝑅 )



def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark
spark = init_spark()

def load_filterd_user_history_1k():
    # 8297836
    # ['index', 'user_id', 'timestamp', 'artist_id', 'artist_name', 'track_id', 'track_name']
    # load top_2_Percent_song history
    history_df = spark.read.parquet("lastfm_dataset/top_20_percent_song_history.parquet")
    # add fractional_play_count
    history_df = history_df.select('user_id','track_id')
    # generate int id for each userID
    userID_map = history_df.select('user_id').drop_duplicates().rdd.zipWithIndex().map(lambda pair:(pair[0][0],pair[1]))
    userID_map_coloumn = ["user_id","id_user"]
    userID_map_df = spark.createDataFrame(data=userID_map, schema=userID_map_coloumn)
    print(userID_map_df.show())
    # generate int id for each trackID
    tracID_map = history_df.select('track_id').drop_duplicates().rdd.zipWithIndex().map(lambda pair:(pair[0][0],pair[1]))
    tracID_map_coloumn = ["track_id","id_track"]
    tracID_map_df = spark.createDataFrame(data=tracID_map, schema=tracID_map_coloumn)

    tracID_map_df.show()

    history_groupby = history_df.groupby('user_id','track_id').count()
    history_groupby_user = history_groupby.groupby('user_id').agg(sf.sum('count').alias('sum'))
    history_groupby = history_groupby.join(history_groupby_user,['user_id'])
    history_groupby = history_groupby.withColumn('fractional_count',col('count')/col('sum'))
    history_groupby = history_groupby.join(userID_map_df,['user_id'])
    history_groupby = history_groupby.join(tracID_map_df,['track_id'])
    groupby_history_df = history_groupby.select('user_id','track_id',"id_user","id_track","count","fractional_count")

    groupby_history_df.show()
    print(groupby_history_df.count())
    # 861902
    print(groupby_history_df.columns)
    # ['user_id', 'track_id', 'id_user', 'id_track', 'count', 'fractional_count']
    print(groupby_history_df.first())
    # +-----------+--------------------+-------+--------+-----+--------------------+
    # |    user_id|            track_id|id_user|id_track|count|    fractional_count|
    # +-----------+--------------------+-------+--------+-----+--------------------+
    # |user_000585|029019ef-c358-4ef...|     91|      44|    1|0.001956947162426...|
    # |user_000792|029019ef-c358-4ef...|    311|      44|   13|0.002452830188679...|
    # groupby_history_df.write.parquet("lastfm_dataset/top_20_Percent_song_fractional_intID_history.parquet")
    return groupby_history_df

def ALS_fractional_count():
    # df = load_filterd_user_history_1k()

    df = spark.read.parquet("lastfm_dataset/top_20_Percent_song_fractional_intID_history.parquet")
    df.show()
    (training, test) = df.randomSplit([0.8, 0.2],seed=100)
    test = test.limit(2000)

    als = ALS(maxIter=5, regParam=0.01, userCol="id_user", itemCol="id_track", ratingCol="fractional_count", implicitPrefs=False,
              coldStartStrategy="drop")
    model = als.fit(training)

    # model.save("model/als_explicit_top20per_song.model")

    # 扩展test set
    # test中有 n tracks，每个user有n条记录，（user1 track1 收听过）（user1 track2 没收听过） 。。。。

    test_full = (
        test.select('id_user').distinct()
            .crossJoin(test.select('id_track').distinct())
            .join(test, on=['id_user', 'id_track'], how='left')
            .fillna(0, subset=['fractional_count'])
            .cache()
    )

    predictions = model.transform(test_full)
    print("user num: %d"%test.select("id_user").distinct().count())
    print("song num: %d"%test.select("id_track").distinct().count())


    listend_song =  test_full.where(col('count') > 0).groupby('id_user').agg(count('*').alias('listened_song'))
    nolistend_song = test_full.where(col('count') == 0).groupby('id_user').agg(count('*').alias('not_listened'))
    listend_song.join(nolistend_song,on=['id_user']).show()

    evaluate(predictions,test_full)


def evaluate(predictions,test_full):
    test1 = predictions.withColumn('rank', row_number().over(Window.partitionBy('id_user').orderBy(desc('prediction'))))
    print("MPR step 1")
    test1.where(col('id_user') == 541).sort(col("rank")).show()
    test1.where(col('id_user') == 181).sort(col("rank")).show()


    n_tracks = test_full.select('id_track').distinct().count()
    # 选出test set 中真的有听得歌，统计这些歌的平均排名.where(col('fractional_count') > 0)
    MPR = predictions.withColumn('rank', row_number().over(Window.partitionBy('id_user').orderBy(desc('prediction')))) \
        .where(col('fractional_count') > 0) \
        .groupby('id_user') \
        .agg(
        count('*').alias('n'),
        sum(1 - col('prediction')).alias('sum_pred'),
        sum(col('rank') / n_tracks).alias('sum_perc_rank'),
        min('rank').alias('min_rank')
    ) \
        .agg(
        (sum('sum_pred') / sum('n')).alias('avg 1-score'),
        (sum('sum_perc_rank') / sum('n')).alias('MPR'),  # the lower the better
        mean(1 / col('min_rank')).alias('MRR')  # the higher the better
    ) \
        .withColumn('MPR*k', col('MPR') * n_tracks) \
        .withColumn('1/MRR', 1 / col('MRR'))

    MPR.show()



def loading_and_evaluate_model():
    df = spark.read.parquet("lastfm_dataset/top_20_Percent_song_fractional_intID_history.parquet")
    (training, test) = df.randomSplit([0.8, 0.2],seed=100)
    model = ALSModel.load("model/als_explicit_top20per_song.model")

    test = test.limit(2000)
    print("test user num: %d"%test.select("id_user").distinct().count())
    print("make recomendation for ecah user from how many songs :%d"%test.select("id_track").distinct().count())

    test_full = (
        test.select('id_user').distinct()
            .crossJoin(test.select('id_track').distinct())
            .join(test, on=['id_user', 'id_track'], how='left')
            .fillna(0, subset=['fractional_count'])
            .cache()
    )

    listend_song =  test_full.where(col('fractional_count') > 0).groupby('id_user').agg(count('*').alias('listened_song'))
    nolistend_song = test_full.where(col('fractional_count') == 0).groupby('id_user').agg(count('*').alias('not_listened'))
    listend_song.join(nolistend_song,['id_user']).show()


    predictions = model.transform(test_full)
    evaluate(predictions,test_full)



# train a new model
ALS_fractional_count()

# loading from model file
# loading_and_evaluate_model()
