from sklearn import datasets

boston = datasets.load_boston()

print(boston.data.shape)
print(boston.feature_names)

'''
linear regression
'''

# Create model object
from sklearn.linear_model import LinearRegression
lr = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)

# Learn
lr.fit(boston.data, boston.target)

# Predcit
pred = lr.predict(boston.data[2].reshape(1, -1))

print(pred)
print(boston.target[2])


# Interpret
print([x for x in zip(boston.feature_names, lr.coef_)])