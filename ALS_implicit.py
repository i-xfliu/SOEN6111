from lastfm_dataset_1K_preprocess import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import pyspark.sql.functions as sf
from pyspark.sql.functions import col
from history_subset import *
from pyspark.ml.recommendation import ALS, ALSModel

# https://colab.research.google.com/drive/1Ugrwtt9uab7PWnAKXuUrerUXuZqNk1no#scrollTo=8ENiRclJJv8k

# 1. loading dataset - "lastfm_dataset/top_20_Percent_song_fractional_intID_history.parquet"
# 5. train ALS model
# 6. parameter : userCol="id_user", itemCol="id_track", ratingCol="count", implicitPrefs=True
# 7. evaluation:  using Mean Percentage Ranking ( 𝑀𝑃𝑅 )  - http://yifanhu.net/PUB/cf.pdf


def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark
spark = init_spark()


def evaluate(predictions,test_full):
    test1 = predictions.withColumn('rank', row_number().over(Window.partitionBy('id_user').orderBy(desc('prediction'))))
    print("MPR step 1")
    # print(test1.count())
    # test1.sort(col("count").desc()).show()
    test1 = test1.select('id_user','id_track','rank','track_id','count')
    test1.where(col('id_user') == 498).sort(col("rank")).show()
    test1.where(col('id_user') == 385).sort(col("rank")).show()
    test1.where(col('id_user') == 193).sort(col("rank")).show()
    test1.where(col('id_user') == 181).sort(col("rank")).show()
    test1.where(col('id_user') == 968).sort(col("rank")).show()
    test1.where(col('id_user') == 39).sort(col("rank")).show()


    n_tracks = test_full.select('id_track').distinct().count()
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


def ALS_trainImplicit():
    # ['user_id', 'track_id', 'count']
    # rdd_user_track_count = load_filterd_user_history_1k()
    # (trainning, validating, test) = rdd_user_track_count.randomSplit([0.6,0.2, 0.2])
    df = spark.read.parquet("lastfm_dataset/top_20_Percent_song_fractional_intID_history.parquet")
    (training, test) = df.randomSplit([0.8, 0.2],seed=100)

    als = ALS(maxIter=5, regParam=0.01, userCol="id_user", itemCol="id_track", ratingCol="count",
              implicitPrefs=True,
              coldStartStrategy="drop")

    model = als.fit(training)

    # model.save("model/als_implicit_top20per_song.model")

    test_full = spark.read.parquet("lastfm_dataset/test_full_5000.parquet")


    print("test user num: %d" % test_full.select("id_user").distinct().count())
    print("make recomendation for ecah user from how many songs :%d" % test_full.select("id_track").distinct().count())

    listend_song = test_full.where(col('fractional_count') > 0).groupby('id_user').agg(
        count('*').alias('listened_song'))
    nolistend_song = test_full.where(col('fractional_count') == 0).groupby('id_user').agg(
        count('*').alias('not_listened'))
    listend_song.join(nolistend_song, ['id_user']).show()

    predictions = model.transform(test_full)
    evaluate(predictions,test_full)


def loading_and_evaluate_model():
    df = spark.read.parquet("lastfm_dataset/top_20_Percent_song_fractional_intID_history.parquet")

    (training, test) = df.randomSplit([0.8, 0.2],seed=100)
    model = ALSModel.load("model/als_implicit_top20per_song.model")

    test_full = spark.read.parquet("lastfm_dataset/test_full_5000.parquet")

    print("test user num: %d" % test_full.select("id_user").distinct().count())
    print("make recomendation for ecah user from how many songs :%d" % test_full.select("id_track").distinct().count())

    listend_song = test_full.where(col('fractional_count') > 0).groupby('id_user').agg(
        count('*').alias('listened_song'))
    nolistend_song = test_full.where(col('fractional_count') == 0).groupby('id_user').agg(
        count('*').alias('not_listened'))
    listend_song = listend_song.join(nolistend_song, ['id_user'])
    listend_song.show()
    listend_song.where(listend_song.id_user.isin(['498', '385', '193', '181','968','39'])).show()

    # predictions = model.transform(test_full).na.drop()
    predictions = model.transform(test_full)
    evaluate(predictions, test_full)





# train a new model
# ALS_trainImplicit()

# loading from model file
loading_and_evaluate_model()
