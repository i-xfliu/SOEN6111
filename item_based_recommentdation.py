from lastfm_dataset_1K_preprocess import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import pyspark.sql.functions as sf
from pyspark.sql.functions import col
from history_subset import *
from pyspark.ml.feature import VectorAssembler, MinHashLSH
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.mllib.linalg import Vectors
from pyspark.ml.linalg import SparseVector
from pyspark.sql.types import FloatType
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances



# 1. loading training subset "lastfm_dataset/top_20_Percent_song_fractional_intID_history.parquet"
# 2. generate user-item matrix
# 3.
class item_based_recommentdation(object):
    spark = init_spark(object)
    df = None
    training = None
    test = None
    train_matrix = None
    test_matrix = None
    model = None


    def init_spark(self):
        spark = SparkSession \
            .builder \
            .appName("Python Spark SQL basic example") \
            .config("spark.some.config.option", "some-value") \
            .getOrCreate()
        return spark


    def generate_user_items_model(self):
        self.df = spark.read.parquet("lastfm_dataset/top_20_Percent_song_fractional_intID_history.parquet")
        (self.training, self.test) = self.df.randomSplit([0.8, 0.2], seed=100)

        self.train_matrix = self.get_matrix(self.training)

        # 根据test set 整理出一个song matrix vector*n
        self.test = spark.read.parquet("lastfm_dataset/test_5000.parquet")
        self.test.groupby('id_user').count().show()

        # self.test_matrix = self.get_matrix(self.test)
        # test 应该从train_matrix中找到相应track的 id
        self.test_matrix = self.train_matrix.join(self.test,['id_track']).select('id_track','features')
        print("test matrix")
        self.test_matrix.show();

        # 1.如果test set 补全在train set中会不会有影响
        brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", bucketLength=2.0,
                                          numHashTables=3)
        self.model = brp.fit(self.train_matrix)




    def get_matrix(self,df):
        limitation = 8
        # pair(id_track,(id_user,count))
        # option1: vector用count构造
        rdd = df.rdd.filter(lambda x: x['count'] > limitation).map(lambda x: (x.id_track, [(x.id_user, x['count'])]))
        # option2: vector用0/1构造
        #rdd = df.rdd.map(lambda x: (x.id_track, [(x.id_user, 1)]))
        # pair(id_track,[(id_user,fractional_count),(id_user,fractional_count)])
        rdd = rdd.reduceByKey(lambda a, b: a + b)
        print(rdd.first())
        # rdd = rdd.map(lambda x: (x[0], SparseVector(len(x[1]), x[1])))
        # 最多有1000个user，设定vector为1000dim，id_user取值范围「0-999」，(id_user,count)即可用来表示SparseVector，第id_user的值为count。
        rdd = rdd.map(lambda x: (x[0], SparseVector(1000, x[1])))
        matrix = spark.createDataFrame(rdd, ['id_track', 'features'])
        return matrix

    # user_df = get_user_items(train_df, train_matrix, id_user)
    def get_user_items(self,history_df, matrix_df, userid):
        user_df = history_df.filter(history_df['id_user'] == userid).select('id_user', 'id_track', 'count')
        matrix_df = matrix_df.join(user_df, ['id_track'])
        matrix_df.show()
        return matrix_df

    def get_test_all_user_items_pair(self, train_df, train_maxtrix, test_df):
        # 选取test集中所有用户的历史数据
        # 选出每个user 在training set的listening history
        # （）
        train_df = train_df.alias("a").join(
            test_df.alias("b"), ['id_user']).select('a.id_user','a.id_track','a.count','a.fractional_count').distinct()

        # 历史数据加上feature
        matrix_df = train_df.join(train_maxtrix, ['id_track'])
        print("get_test_all_user_items_pair")
        matrix_df.show()
        # +--------+-------+-----+--------------------+
        # |id_track|id_user|count|            features|
        # +--------+-------+-----+--------------------+
        # |      29|    830|   12|(1000,[30,72,73,8...|
        return matrix_df


    def get_score(self,sim_df, train_df, id_user):
        # sim_df : id,sim
        id_track_list = sim_df.select("id_track").rdd.flatMap(lambda x: x).collect()
        user_track_df = train_df.filter((train_df.id_user == id_user) & (col('id_track').isin(id_track_list)))
        full_df = sim_df.join(user_track_df, ['id_track'], how='left')
        score = full_df.withColumn('score', 1/col('distCol') * col('count')).agg(avg(col('score')))
        # user_track_df.show()
        # full_df.show()
        # score.show()
        score = score.rdd.flatMap(lambda x: x).collect()[0]
        # print(score)
        return score


    def get_predcit_single_user(self, id_user, train_df, train_matrix, test_matrix):

        user_df = self.get_user_items(train_df, train_matrix, id_user)
        similar = self.model.approxSimilarityJoin(user_df, test_matrix, 1000, "JaccardDistance")
        items_similarity = pairwise_distances(train_matrix.T, metric='cosine')
        similar = similar.select(col('datasetA.id_user').alias('id_user'),
                                 col('datasetA.id_track').alias('user_listen'),
                                 col('datasetA.count').alias('count'),
                                 col('datasetB.id_track').alias('test_song'),
                                 col('distCol')).where(col('user_listen') != col('test_song'))
        window = Window.partitionBy('test_song').orderBy('distCol')
        # select top k similar songs
        similar = similar.select('*', rank().over(window).alias('rank')).filter(col('rank') <= 5)
        # id_user|user_listen|count|test_song|           distCol|rank|
        #      19|      20608|    1|     4590|               7.0|   1|
        #      19|      20927|    4|     4590| 8.306623862918075|   2|
        return similar

        similar.show()



    def get_predcit_multiple_users(self):
        # train_df, train_matrix, test_matrix, test
        # computer the similarity for all of test track for each user on one go
        all_user_history_df = self.get_test_all_user_items_pair(self.training, self.train_matrix, self.test)
        track_num = self.test.select('id_track').distinct().count()
        user_num = self.test.select('id_user').distinct().count()
        print("track_num: %d" % track_num)
        print("user_num: %d" % user_num)


        similar = self.model.approxSimilarityJoin(all_user_history_df, self.test_matrix, 1000, distCol="JaccardDistance")

        similar = similar.select(col('datasetA.id_user').alias('id_user'),
                                 col('datasetA.id_track').alias('user_listen'),
                                 col('datasetA.count').alias('count'),
                                 col('datasetA.fractional_count').alias('fractional_count'),
                                 col('datasetB.id_track').alias('test_song'),
                                 # col('distCol')
                                 col('JaccardDistance').alias('distCol')
                                 ).where(col('user_listen') != col('test_song'))

        # similar.write.parquet("model/similar_matrix_5000test_jaccard.parquet")

        window = Window.partitionBy('id_user', 'test_song').orderBy('distCol')
        similar = similar.select('*', rank().over(window).alias('rank')).filter(col('rank') <= 5)
        # id_user|user_listen|count|test_song|           distCol|rank|
        #      19|      20608|    1|     4590|               7.0|   1|
        #      19|      20927|    4|     4590| 8.306623862918075|   2|
        print(similar.count())

        similar.show()
        score_df = similar.withColumn('similar',  (1-col('distCol')))
        score_df = score_df.withColumn('score_temp', col('similar') * col('count'))
        score_df = score_df.groupby('id_user', 'test_song').agg(sum('score_temp').alias('sum_score'),sum((col('similar'))).alias('sum_similar'))
        score_df = score_df.withColumn('score',  col('sum_score') / col('sum_similar'))
        score_df.show()
        print(score_df.count())

        score_df.sort('id_user').show()
        self.evaluate(score_df, self.test)

        return similar


    def evaluate(self,predictions, test_df):
        # predictions mark the the songs not be listened
        rank = predictions.withColumn('rank', row_number().over(Window.partitionBy('id_user').orderBy(desc('score'))))

        cond = [rank.id_user == test_df.id_user, rank.test_song == test_df.id_track]
        # 展开 test df,
        predictions = rank.join(test_df, cond, how='left').drop(test_df.id_user).fillna(0)


        print("MPR step 1")
        print(predictions.count())
        # test1.filter(test1['id_user'] == 26).show()
        listend_song = predictions.where(col('count') > 0).groupby('id_user').agg(count('*').alias('listened_song'))
        nolistend_song = predictions.where(col('count') == 0).groupby('id_user').agg(count('*').alias('not_listened'))
        listend_song.join(nolistend_song, on=['id_user']).show()

        # predictions.where(col('id_user') == 498).sort(col("rank")).show()
        # predictions.where(col('id_user') == 754).sort(col("rank")).show()
        # predictions.where(col('id_user') == 443).sort(col("rank")).show()
        # predictions.where(col('id_user') == 304).sort(col("rank")).show()
        # predictions.where(col('id_user') == 871).sort(col("rank")).show()
        # predictions.where(col('id_user') == 181).sort(col("rank")).show()
        # predictions.where(col('id_user') == 621).sort(col("rank")).show()

        n_tracks = test_df.select('id_track').distinct().count()
        # predictions.withColumn('rank', row_number().over(Window.partitionBy('id_user').orderBy(desc('score'))))

        MPR = predictions\
            .where(col('fractional_count') > 0) \
            .groupby('id_user') \
            .agg(
            count('*').alias('n'),
            sum(1 - col('score')).alias('sum_pred'),
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


        predictions = predictions.select('id_user', 'test_song', 'rank', 'track_id', 'count')
        predictions.where(col('id_user') == 498).sort(col("rank")).show()
        predictions.where(col('id_user') == 385).sort(col("rank")).show()
        predictions.where(col('id_user') == 193).sort(col("rank")).show()
        predictions.where(col('id_user') == 181).sort(col("rank")).show()
        predictions.where(col('id_user') == 968).sort(col("rank")).show()
        predictions.where(col('id_user') == 39).sort(col("rank")).show()



    def load_and_evalaute_similar_matrix(self):

        similar = spark.read.parquet("model/similar_matrix_5000test_jaccard.parquet")

        window = Window.partitionBy('id_user', 'test_song').orderBy('distCol')

        similar = similar.select('*', rank().over(window).alias('rank')).filter(col('rank') <= 5)
        # id_user|user_listen|count|test_song|           distCol|rank|
        #      19|      20608|    1|     4590|               7.0|   1|
        #      19|      20927|    4|     4590| 8.306623862918075|   2|
        print(similar.count())
        similar.show()
        score_df = similar.withColumn('similar',  (1-col('distCol')))
        score_df = score_df.withColumn('score_temp', col('similar') * col('count'))
        score_df = score_df.groupby('id_user', 'test_song').agg(sum('score_temp').alias('sum_score'),sum((col('similar'))).alias('sum_similar'))
        score_df = score_df.withColumn('score',  col('sum_score') / col('sum_similar'))
        score_df.show()
        print(score_df.count())

        score_df.sort('id_user').show()
        self.evaluate(score_df, self.test)




model = item_based_recommentdation()

model.generate_user_items_model()

# # train a new model
# model.get_predcit_multiple_users()

# loading from model file
model.load_and_evalaute_similar_matrix()
