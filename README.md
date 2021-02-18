# Don't Stop the Music

## Introduction

The recommender system has many successful applications in the industry. According to statistics, 40% of Amazon's sales are generated by the recommender system; 75% of Netflix users use the recommendation system to find their favorite videos; 30% of Netflix users use keywords to search the products they need before online shopping. At present, almost all news, search, advertising, short video applications are based on the recommender system.

Our study is based on the last.fm dataset in Million Song Dataset  and Last.fm Dataset (1K users) , aiming to make efficient offline music recommendations for users.  The last.fm dataset in Million Song Dataset contains songs metadata and song-level tags, and the Last.fm Dataset (1K users) contains 1k users' listening history and users profile. 

Before implementing the recommendation algorithm, we can explore the data distribution in the datasets. for example, the top k most popular songs, artists, tags, and as on,  then use MatLab to visualize then.

In this project, we plan to set up a music recommender system. Specifically, a system recommends music that the listeners would like to hear no matter they heard before or not. Our objectives include 

1) Data processing: 

The data is processed in the offline layer, i.e., using spark and other computing engines for data analysis and feature extraction. 

2) Algorithms implementation: 

We will use the KNN Item-based collaborative filtering, due to our users dataset is a listening history which is implicit feedback, thus for this algorithm, we need to convert this dataset to an explicit feedback dataset. This operation may impact the effect of the recommendation.For this reason, we will try the other algorithm Matrix factorization-based collaborative filtering, which can directly use user-item interaction matrix as training data.

3) Evaluation: 

We will use some metrics to elevalute the models, such as recall, precise, mAP and AUC.



## Materials and Methods

### DataSet

Our datasets come from last.fm. There are two parts of dataset, 

1. The last.fm dataset in Million Song Dataset

   http://millionsongdataset.com/lastfm/ , which provides songs metadata and song-level tags datasets.

   1) track_metadata.db

   Which contains 100k songs with track_id, title text, song_id , release time, artist_id , artist_mbid, artist_name, duration, artist_familiarit , artist_hotttnesss  year.

   2) lastfm_tags.db

   which provides a list of unique tags and a list of song-tag pairs. There are 50K unique tags and each song can have multiple tags. Now, we can get song-level data, used as item dataset  from the above two database.

2. Last.fm Dataset - 1K users

   http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html, which provides users' listening history  and  users' brief profile collected from last.fm.  The listening history contains almost 2000k lines with <index, user_id, timestamp, artist_id, artist_name, track_id, track_name> tuple, in which has almost 1k users. The user profile data contains 1k users with <userid, gender, age, country, signup time>.



### Data Analysis

#### clean data

The listening history dataset is very huge, has 2000k records, which contains a large number of not useful data. So the first step is to extract useful data, intersection the listening history and songs dataset, thus we can just keep the history that has responding songs in the songs dataset. After the intersect operation, the listening history dataset is reduced to 800k lines and the songs dataset is reduced to 20k lines.

#### The statistics

Before we starting the algorithm, we can have a look at the dataset distribution. For songs, we can statistic the top k most popular songs, artists, tags, countries and, so on. For users, we can statistic the gender proportions and aged distribution. we can use the MatLab library to visualize those statistic results.

#### Recommendation Algorithm

1. ##### KNN Item based collaborative filtering 

   For implementing item-based collaborative filtering, we need two kinds of data, users feature and items feature. Thus we need feature engineering, our plan is to extract song features and user features from the song-tag dataset and user listening history. This is a very preliminary method that will lose the listening sequence information. (这段数据可能有点问题)

   When data is ready, we then build song-user matrix and use sklearn.neighbors.NearestNeighbors to train the model.

2. ##### Matrix factorization-based collaborative filtering

   The users listening history is a user behavior dataset, which doesn't explicitly reflect the taste of users, thus it's called implicit feedback. If we use the statistic method to construct user preferences, can lose some information contained in listening history. 
   For implicit feedback, we can use matrix factorization-based collaborative filtering to implement a recommendations system.

   We can use spark.mllib.recommendation.ALS model to handle implicit feedback dataset, the model can find latent factors in the listening history dataset.

3. ##### Evaluation

   We will use some metrics to elevalute the models, such as recall, precise, mAP and AUC.







