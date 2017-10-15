# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 22:10:22 2017

@author: whuang67
"""

import numpy as np

class LinearRegression():
    def __init__(self, step=3000, learning_rate=.01):
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
            y_pred = np.where(np.matmul(self.X, self.w) > 0,
                              np.ones(self.X.shape[0]),
                              np.zeros(self.X.shape[0]))
            gradient = - np.matmul(self.X.T, self.y - y_pred)
            # gradient = -(self.X.shape[0]-np.dot(self.y, y_pred)) * np.matmul(self.X.T, self.y)
            self.w -= self.learning_rate * gradient
            print(self.w)
    
    def predict(self, test_data = None):
        try:
            if test_data is None:
                test_data = self.X
        
            output = np.matmul(test_data, self.w)
            return np.where(output > 0,
                            np.ones(test_data.shape[0]),
                            np.zeros(test_data.shape[0]))
            
        except AttributeError:
            print("Please fit first!")

def get_accuracy(y, y_pred):
    return np.mean(y == y_pred)

if __name__ == "__main__":
    np.random.seed(1)
    P = 4; N = 200; rho = .5
    V = rho**abs(np.array([[i-j for i in range(1, P+1)] for j in range(1, P+1)]))
    X = np.random.multivariate_normal(mean = np.zeros(P), cov = V, size = N)
    beta = np.array([1, 1, .5, .5])
    Y = np.matmul(X, beta.T) + np.random.normal(size = N)
    Y_ = np.where(Y > 0, np.ones(N), np.zeros(N))
    
    ## Model fitting
    model = LinearRegression()
    model.fit(X, Y_)
    Y_pred = model.predict()
    print(model.w)
    print(get_accuracy(Y_, Y_pred))
