"""
datasets:
- movie lens 100k
    - u.user: user_id|age|gender|job|zip_no
    - u.item: movie_id|movie_title|open_date|video_date|imdb_url|genre_1_hot..
    - u.data: user_id\tmove_id\trating\trating_time
"""

'''
Load datasets
'''
import codecs
import os


def read_data(fin, delim):
    info_li = []

    for line in codecs.open(fin, 'r', encoding='latin-1'):

        line_items = line.strip().split(delim)

        key = int(line_items[0])

        if (len(info_li) + 1) != key:
            print('errors at data_id')
            exit(0)

        info_li.append(line_items[1:])

    print('rows in %s: %d' % (fin, len(info_li)))
    return info_li


dataset_root_dir_path = '/home/socurites/Backup/Data/MoveLens/ml-100k'
fin_user = os.path.join(dataset_root_dir_path, 'u.user')
fin_movie = os.path.join(dataset_root_dir_path, 'u.item')
fin_data = os.path.join(dataset_root_dir_path, 'u.data')

user_info_dic = read_data(fin_user, '|')
movie_info_dic = read_data(fin_movie, '|')


import numpy as np

# utility matrix R
R = np.zeros((len(user_info_dic), len(movie_info_dic)), dtype=np.float64)

for line in codecs.open(fin_data, 'r', encoding='latin-1'):
    user, movie, rating, date = line.strip().split('\t')

    R[int(user)-1, int(movie)-1] = float(rating)

print("user %d rated movie %d as %f" % (1, 11, R[0, 10]))


'''
Inspect datasets
'''
from scipy import stats

# user basic stats
user_mean_li = []
for i in range(0, R.shape[0]):
    user_rating = [x for x in R[i] if x > 0.0]
    user_mean_li.append(stats.describe(user_rating).mean)

print(stats.describe(user_mean_li))


# movie basic stats
movie_mean_li = []
for i in range(0, R.shape[1]):
    R_T = R.T
    movie_rating = [x for x in R_T[i] if x > 0.0]
    movie_mean_li.append(stats.describe(movie_rating).mean)

print(stats.describe(movie_mean_li))