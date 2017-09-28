# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 23:52:52 2017

@author: whuang67
"""

import numpy as np

# Question 1
P, N, rho = 4, 200, 0.5
V = np.array([[1, .5, .25, .125],
              [.5, 1, .5, .25],
              [.25, .5, 1, .5],
              [.125, .25, .5, 1]])


np.random.seed(1)
X = np.random.multivariate_normal(np.repeat(0, P), V, N)
beta = np.array([1, 1, .5, .5])
y = np.matmul(X, beta) + np.random.normal(0, 1, N)

### a
sigma = np.cov(X.T)
print(sigma)

### b
def mydist(arr_1, arr_2):
    return (np.sum((arr_1 - arr_2)**2))**.5

dist = np.repeat(0., len(X))
for i, point in enumerate(X):
    dist[i] = mydist(point, np.array([.5, .5, .5, .5]))



### c
def mydist2(arr_1, arr_2, s):
    return np.matmul(np.matmul(np.transpose(arr_1-arr_2), np.linalg.inv(s)), 
                    (arr_1-arr_2))

dist2 = np.repeat(0., len(X))
for i, point in enumerate(X):
    dist2[i] = mydist2(point, np.array([.5, .5, .5, .5]), sigma)

### d

# Question 2
### a
def degrees_of_freedom(y_true, y_pred, sigma = 1.):
    return np.sum(np.diag(np.cov(y_pred, y_true)))/sigma
    # return np.cov(y_pred, y_true).trace()/sigma

### b
# Generate X
X = np.random.multivariate_normal([0., 0., 0., 0.],
                                  [[1., 0., 0., 0.],
                                   [0., 1., 0., 0.],
                                   [0., 0., 1., 0.],
                                   [0., 0., 0., 1.]],
                                   200)

# True function f(X)
f_X = np.matmul(X, np.array([1., 2., 3., 4.]))

# Add noise
np.random.seed(1)
y = f_X + np.random.normal(0., 1., 200)

# Fit knn and make prediction
from sklearn.neighbors import KNeighborsRegressor
reg = KNeighborsRegressor().fit(X, y)
y_pred = reg.predict(X)

# Degrees of freedom
print(degrees_of_freedom(y, y_pred))

# Repeat 20 times to get a better fit
df_arr = np.repeat(0., 20)
np.random.seed(1)
for i in range(20):
    y = f_X + np.random.normal(0., 1., 200)
    reg = KNeighborsRegressor().fit(X, y)
    y_pred = reg.predict(X)
    df_arr[i] = degrees_of_freedom(y, y_pred)
print(df_arr.mean())

### c
y_pred_lr = np.matmul(
        np.matmul(np.matmul(X, np.linalg.inv(np.matmul(X.T, X))), X.T), y)
df_lr = np.cov(y_pred_lr, y).trace()/1.
print(df_lr)


# Question 3
import os
os.chdir("C:/users/whuang67/downloads")
import pandas as pd
SAheart = pd.read_csv("SAheart.csv")
X = SAheart[["age", "tobacco"]]
y = SAheart["chd"]

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
parameter = {"n_neighbors": range(3, 31, 2)}
reg = GridSearchCV(KNeighborsClassifier(),
                   parameter,
                   scoring = "accuracy",
                   cv = 10)
reg.fit(X, y)

cv_result = reg.cv_results_
import matplotlib.pyplot as plt

plt.plot(cv_result["mean_test_score"], label = "CV Error")
plt.plot(cv_result["mean_train_score"], label = "Training Error")
plt.xticks(range(14), range(3, 31, 2))
plt.legend(); plt.xlabel("Neighoors"); plt.ylabel("Accuracy")
plt.title("10-Fold Cross-Validation")
plt.show()