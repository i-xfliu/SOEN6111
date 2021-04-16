from history_subset import *
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np


def most_popular_songs():
    top_20_songs = load_most_popular_songs_set(15)
    # 'track_id','track_name','count'
    track_name_list = list(top_20_songs.select('track_name').toPandas()['track_name'])
    print(type(track_name_list))
    print(len(track_name_list))
    y_pos = np.arange(len(track_name_list))
    print(y_pos)
    count_list = list(top_20_songs.select('count').toPandas()['count'])
    print(count_list)
    plt.bar(y_pos, count_list, align='center', alpha=0.5)
    plt.xticks(y_pos, track_name_list, rotation='vertical')
    plt.ylabel('Play count')
    plt.title('Most popular songs')
    plt.show()

# Most active users
def most_active_users():
    top_20_users = load_most_active_users(20)
    # 'user_id','count'
    user_id_list = list(top_20_users.select('user_id').toPandas()['user_id'])
    print(type(user_id_list))
    print(len(user_id_list))
    y_pos = np.arange(len(user_id_list))
    print(y_pos)
    count_list = list(top_20_users.select('count').toPandas()['count'])
    print(count_list)
    plt.bar(y_pos, count_list, align='center', alpha=0.5)
    plt.xticks(y_pos, user_id_list, rotation='vertical')
    plt.ylabel('Play count')
    plt.title('Most active users')
    plt.show()

def user_play_count_distribution():
    users_placy_df = load_users_placy_count().toPandas()
    users_placy_df.columns = ['user_id', 'play_count']
    users_placy_df = users_placy_df.loc[(users_placy_df['play_count'] > 10) & (users_placy_df['play_count'] <= 20000)]
    fig = plt.figure(figsize=(10, 5))
    bins = np.arange(users_placy_df.play_count.min(), users_placy_df.play_count.max(), 500)
    users_placy_df.play_count.plot(kind='hist', bins=bins)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off')  # ticks along the top edge are off) # labels along the bottom edge are off
    plt.ylabel("user count")
    plt.xlabel("user play count")
    plt.title("User play count Distribution")
    plt.show()

def track_play_count_distribution():
    tracks_placy_df = load_tracks_placy_count().toPandas()
    tracks_placy_df.columns = ['track_id', 'play_count']
    tracks_placy_df = tracks_placy_df.loc[(tracks_placy_df['play_count'] > 500)& (tracks_placy_df['play_count'] <= 6000)]
    fig = plt.figure(figsize=(10, 5))
    bins = np.arange(tracks_placy_df.play_count.min(), tracks_placy_df.play_count.max(), 100)
    tracks_placy_df.play_count.plot(kind='hist', bins=bins)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off')  # ticks along the top edge are off) # labels along the bottom edge are off
    plt.ylabel("track count")
    plt.xlabel("track play count")
    plt.title("Track play count Distribution")
    plt.show()

# most_popular_songs()
# most_active_users()
user_play_count_distribution()
# track_play_count_distribution()