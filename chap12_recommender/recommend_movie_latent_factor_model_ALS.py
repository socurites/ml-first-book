"""
Recommend movie based on trained model
"""

import numpy as np
from eval_cf_latent_factor_modeL_ALS import R, R_hat, movie_info_dic

# make min_value to zero
R_hat -= np.min(R_hat)

# make max_value to 5
R_hat *= float(5) / np.max(R_hat)


def recommend_by_user(user):
    user_seen_movies = sorted(list(enumerate(R_hat[user])), key=lambda x:x[1], reverse=True)
    print(user_seen_movies)

    recommended = 1
    for movie_info in user_seen_movies:
        if R[user][movie_info[0]] == 0:
            movie_title = movie_info_dic[movie_info[0] + 1]
            movie_score = movie_info[1]

            print("rank %d recommendation: %s(%.3f)" % (recommended, movie_title[0], movie_score))

            recommended += 1
            if recommended == 6:
                break


user = 0
recommend_by_user(user)