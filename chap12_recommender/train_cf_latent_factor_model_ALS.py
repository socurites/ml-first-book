"""
Train CF(Collaborative Filtering) by ALS
"""

from numpy.ma import mean

from preprocess_data import R

'''
Initalize X, Y
'''
K = 100

import numpy as np
m, n = R.shape

X = 5 * np.random.rand(m, K)
Y = 5 * np.random.rand(K, n)


'''
To minimize L-2 regularized loss,
X <- X that satisfies (Y*Y.T + lambda*I)X.T = Y*R.T
Y <- Y that satisfies (X.T*X + lambda*I)Y = X.T*R
'''

from sklearn.metrics import mean_squared_error

n_iter = 20         # num of iteration
lambda_ = 0.1       # regularization parameter
errors = []         # errors per iteration

for i in range(0, n_iter):
    X = np.linalg.solve(np.dot(Y, Y.T) + lambda_ * np.eye(K), np.dot(Y, R.T)).T
    Y = np.linalg.solve(np.dot(X.T, X) + lambda_ * np.eye(K), np.dot(X.T, R))

    if i % 10 == 0:
        print("iteration %d is completed" % (i))

    error = mean_squared_error(R, np.dot(X, Y))
    errors.append(error)
    print("error at iteration %d: %.4f" % (i, error))


R_hat = np.dot(X, Y)

print('Error of rated moveis: %.4f' % (mean_squared_error(R, R_hat)))

