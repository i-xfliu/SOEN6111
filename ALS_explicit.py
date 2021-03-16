from lastfm_dataset_1K_preprocess import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row


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
    history_df = load_lastfm_1k().load_useful_history()
    history_df = history_df.limit(20000)
    history_df = history_df.drop('index','timestamp', 'artist_id', 'artist_name','track_name')
    # generate int id for each userID
    userID_map = history_df.select('user_id').drop_duplicates().rdd.zipWithIndex().map(lambda pair:(pair[0][0],pair[1]))
    userID_map_coloumn = ["user_id","id_user"]
    userID_map_df = spark.createDataFrame(data=userID_map, schema=userID_map_coloumn)
    print(userID_map_df.show())
    # generate int id for each trackID
    tracID_map = history_df.select('track_id').drop_duplicates().rdd.zipWithIndex().map(lambda pair:(pair[0][0],pair[1]))
    tracID_map_coloumn = ["track_id","id_track"]
    tracID_map_df = spark.createDataFrame(data=tracID_map, schema=tracID_map_coloumn)
    print(tracID_map_df.show())

    history_groupby = history_df.groupby('user_id','track_id').count()
    print(history_groupby.show())
    history_groupby = history_groupby.join(userID_map_df,history_groupby.user_id == userID_map_df.user_id)
    history_groupby = history_groupby.join(tracID_map_df, history_groupby.track_id == tracID_map_df.track_id)
    groupby_history_df = history_groupby.select("id_user","id_track","count")

    print(groupby_history_df.show())
    print(groupby_history_df.count())
    # 1212114
    print(groupby_history_df.columns)
    # ['user_id', 'track_id', 'count']
    print(groupby_history_df.first())
    # Row(user_id='user_000162', track_id='e382ca9a-113d-4dc6-8a05-d236a126c81a', count=3)
    return groupby_history_df

def ALS_count():
    df = load_filterd_user_history_1k()
    (training, test) = df.randomSplit([0.8, 0.2])
    # 加normalize


    als = ALS(maxIter=5, regParam=0.01, userCol="id_user", itemCol="id_track", ratingCol="count", implicitPrefs=False,
              coldStartStrategy="drop")

    model = als.fit(training)

    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="count",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))

    # Generate top 10 movie recommendations for each user
    userRecs = model.recommendForAllUsers(10)
    # Generate top 10 user recommendations for each movie
    songRecs = model.recommendForAllItems(10)


ALS_count()




