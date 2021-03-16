from history_subset import *
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np


def most_popular_songs():
    top_20_songs = load_most_popular_songs_set(20)
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

# Most Popular Artist


most_popular_songs()