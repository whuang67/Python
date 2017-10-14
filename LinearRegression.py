# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 22:10:22 2017

@author: whuang67
"""

import numpy as np

class LinearRegression():
    def __init__(self, step=10, learning_rate=.01):
        self.step = step
        self.learning_rate = learning_rate
    
    def fit(self, X, y, intercept=True):
        self.intercept = intercept
        
        if self.intercept:
            self.X = np.hstack((np.ones((X.shape[0], 0)), X))
        else:
            self.X = X
        self.y = y

        self.w = np.ones(self.X.shape[1])
        for _ in range(self.step):
            y_pred = np.matmul(self.X, self.w)
            self.w += self.learning_rate * np.matmul(self.X.T, self.y-y_pred)
            print(self.w)
    
    def predict(self, test_data = None):
        try:
            if test_data is None:
                test_data = self.X
        
            return np.matmul(test_data, self.w)
        except AttributeError:
            print("Please fit first!")




if __name__ == "__main__":
    np.random.seed(1)
    P = 4; N = 200; rho = .5
    V = rho**abs(np.array([[i-j for i in range(1, P+1)] for j in range(1, P+1)]))
    X = np.random.multivariate_normal(mean = np.zeros(P), cov = V, size = N)
    beta = np.array([1, 1, .5, .5])
    Y = np.matmul(X, beta.T) + np.random.normal(size = N)
    Y_ = np.where(Y > 0, np.ones(N), np.zeros(N))
    
    model = LinearRegression()
    model.fit(X, Y)
    print(model.w)