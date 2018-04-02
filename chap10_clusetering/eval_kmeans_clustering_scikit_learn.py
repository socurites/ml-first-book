from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

from train_kmeas_clustering_scikit_learn import test_data, train_data

'''
Eval using silhouette coefficient
'''
for k in range(2, 9):
    km = KMeans(n_clusters=k).fit(test_data)

    print("score for %d culsters: %.3f" % (k, silhouette_score(test_data, km.labels_)))


'''
Eval using elbow method
'''

ssw_dic = {}

for k in range(1, 8):
    km = KMeans(n_clusters=k).fit(test_data)
    ssw_dic[k] = km.inertia_


from matplotlib import pyplot as plt

plot_data_x = list(ssw_dic.keys())
plot_data_y = list(ssw_dic.values())
plt.xlabel("# of clusters")
plt.ylabel("within ss")
plt.plot(plot_data_x, plot_data_y, linestyle="-", marker='o')
plt.show()