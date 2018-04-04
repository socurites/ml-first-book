"""
Evaluate trained model by test datasets
"""

from preprocess_data import R, movie_info_dic
import numpy as np

'''
Split dataset into train / test
'''
train = R.copy()
test = np.zeros(R.shape)

n_test = 10

for user in range(R.shape[0]):
    # choose indexes of nonzero ratings of a user
    # print(R[0, :].nonzero()[0])
    test_index = np.random.choice(R[user, :].nonzero()[0], size=n_test, replace=False)

    train[user, test_index] = 0
    test[user, test_index] = R[user, test_index]


'''
Define training process as a function
'''
def compute_ALS(D):
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
        X = np.linalg.solve(np.dot(Y, Y.T) + lambda_ * np.eye(K), np.dot(Y, D.T)).T
        Y = np.linalg.solve(np.dot(X.T, X) + lambda_ * np.eye(K), np.dot(X.T, D))

        if i % 10 == 0:
            print("iteration %d is completed" % (i))

        #error = mean_squared_error(D, np.dot(X, Y))
        error = mean_squared_error(D[D.nonzero()], np.dot(X, Y)[D.nonzero()])
        errors.append(error)
        print("error at iteration %d: %.4f" % (i, error))


    R_hat = np.dot(X, Y)

    print('Error of rated moveis: %.4f' % (mean_squared_error(R, R_hat)))

    return(R_hat, errors)


R_hat, train_errors = compute_ALS(train)

_, test_errors = compute_ALS(test)


'''
Plot train/test errors
'''
from matplotlib import pyplot as plt

x = range(0,20)

plt.xlim(0, 20)
plt.ylim(0, 15)

plt.xlabel('iteration')
plt.ylabel('MSE')

plt.xticks(x, range(0,20))

test_plot = plt.plot(x, test_errors, '--', label='test error')
train_plot = plt.plot(x, train_errors, label='train error')

#plt.legend(handles=[train_plot, test_plot])

plt.show()