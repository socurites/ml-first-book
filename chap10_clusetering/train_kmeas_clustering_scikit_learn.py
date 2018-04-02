import random
from preprocess_data import user_product_vec_li
from sklearn.cluster import KMeans

'''
Split data into train/test
'''
random.shuffle(user_product_vec_li)

train_data = user_product_vec_li[:2500]
test_data = user_product_vec_li[2500:]

print("# of train data: %d, # of test data: %d" % (len(train_data), len(test_data)))


'''
Create model
'''
km_predict = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=20)


'''
Train
'''
km_predict.fit(train_data)


'''
Test
'''
km_predict_result = km_predict.predict(test_data)
print(km_predict_result)