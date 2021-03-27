import sys
from collections import defaultdict
from itertools import combinations
import numpy as np
import random
import csv
import pdb
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession


def pairbyuser(line):
    return line[0],(line[1],float(line[2]))

# get all the track pairs for each user
def gettrackpairs(track_and_rating):

    # [('2', 3.0), ('3', 1.0), ('5', 2.0), ('9', 4.0), ('11', 1.0), ('12', 2.0), ('15', 1.0), ('17', 1.0), ('19', 1.0),
    #  ('21', 1.0), ('23', 1.0), ('26', 3.0), ('27', 1.0), ('28', 1.0), ('29', 1.0), ('30', 1.0), ('31', 1.0),
    #  ('34', 1.0), ('37', 1.0), ('41', 2.0), ('44', 1.0), ('45', 2.0), ('46', 1.0), ('47', 1.0), ('48', 1.0),
    #  ('50', 1.0), ('51', 1.0), ('54', 1.0), ('55', 1.0), ('59', 2.0), ('61', 2.0), ('64', 1.0), ('67', 1.0),
    #  ('68', 1.0), ('69', 1.0), ('71', 1.0), ('72', 1.0), ('77', 2.0), ('79', 1.0), ('83', 1.0), ('87', 1.0),
    #  ('89', 2.0), ('91', 3.0), ('92', 4.0), ('94', 1.0), ('95', 2.0), ('96', 1.0), ('98', 1.0), ('99', 1.0)]
    pair_list = []
    for track1,track2 in combinations(track_and_rating, 2):
        pair_list.append(((track1[0], track2[0]),(track1[1], track2[1])))
    return pair_list



def findUserPairs(item_id,users_with_rating):
    '''
    For each movie, find all user with same movie
    '''
    for user1,user2 in combinations(users_with_rating,2):
        return (user1[0],user2[0]),(user1[1],user2[1])

def combine_item_similarity(track_pair, track_sim_data):

    (track1_id,track2_id) = track_pair
    return track1_id,(track2_id,track_sim_data)

def cosineSimilarity(track_pair, rating_pairs):
    sum_x, sum_xy, sum_y= (0.0, 0.0, 0.0)

    for rating_pair in rating_pairs:
        sum_x += np.float(rating_pair[0]) * np.float(rating_pair[0])
        sum_y += np.float(rating_pair[1]) * np.float(rating_pair[1])
        sum_xy += np.float(rating_pair[0]) * np.float(rating_pair[1])

    cosine_sim = cosine(sum_xy, np.sqrt(sum_x), np.sqrt(sum_y))
    return track_pair, cosine_sim

def cosine(dot_product,rating1_norm_squared,rating2_norm_squared):
    '''
    The cosine between two vectors
    '''
    num = dot_product
    den = rating1_norm_squared * rating2_norm_squared
    return (num / (float(den))) if den else 0.0

def nearNeighbors(movie_id, movie_and_sims, number):
    '''
    Sort the movie predictions list by similarity and select the top N related users
    '''
    movie_and_sims.sort(key = lambda x: x[1],reverse=True)
    return movie_id, movie_and_sims[:number]

def topMusicRecommendations(user_id, track_and_rating, dict_similar_track, n):
    '''
    Calculate the top N movie recommendations for each user using the
    weighted sum approach
    '''

    # initialize dicts to store the score of each individual movie and movie can exist with more than one movie
    t = defaultdict(int)
    sim_sum = defaultdict(int)

    for (track, rating) in track_and_rating:

        # get the nearst k tracks from the dict
        list_near_tracks = dict_similar_track.get(track,None)

        if list_near_tracks:
            for (near_track,(sim,count)) in list_near_tracks:
                if near_track != track:

                    # update totals and sim_sum
                    t[near_track] += sim * rating
                    sim_sum[near_track] += sim

    # create the normalized list of scored movies
    scored_track = [(total/sim_sum[track],track) for track,total in t.items()]

    # sort the scored movies in ascending order
    scored_track.sort(reverse=True)


    # ranked_items = [x[1] for x in scored_items]

    return user_id,scored_track[:n]

def toCSVLine(data):
  return ','.join(str(d) for d in data)

limit = 5 # limit to be selected
k = 10 # k for knn
def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark

def count_iterable(i):
    return sum(1 for e in i)

def pairlist(list):
    return list[0],list[1]

def item_based_process(spark, parts):
    # why pair by user? to ensure the track was listened by the same user
    user_pairs = parts.map(pairbyuser).groupByKey()
    # two track id as key, and rating pairs as value
    track_pairs_final = user_pairs.filter(lambda pair: count_iterable(pair[1]) > 1).flatMap(
        lambda pair: gettrackpairs(pair[1])).groupByKey()

    # track_pairs_final = dff.map(lambda x: pairlist(x)).groupByKey()
    # after groupByKey, the value is all the rating pairs from users who listened the two tracks(two track id is the key)
    # if common user number between the track pair less than limit, ignore
    track_sim = track_pairs_final.filter(lambda pair: count_iterable(pair[1]) > limit).map(
        lambda pair: cosineSimilarity(pair[0], pair[1]))
    #
    track_sim_finally = track_sim.map(
        lambda pair: combine_item_similarity(pair[0], pair[1])).groupByKey() \
        .map(lambda x: (x[0], list(x[1])))

    list_track_sim_knn = track_sim_finally.map(
        lambda pair: nearNeighbors(pair[0], pair[1], k)).collect()

    dict_similar_track = {}
    for (track, data) in list_track_sim_knn:
        dict_similar_track[track] = data
    i = spark.sparkContext.broadcast(dict_similar_track)
    user_music_rec = user_pairs.map(
        lambda p: topMusicRecommendations(p[0], p[1], i.value, 3)).collect()




def test():
    spark = init_spark()
    lines = spark.read.text('./sample_movielens_ratings.txt').rdd
    parts = lines.map(lambda row: row.value.split("::"))
    item_based_process(spark, parts)

