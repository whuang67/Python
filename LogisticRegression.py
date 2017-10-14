# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 21:20:39 2017

@author: whuang67
"""

import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def log_likelihood(feature, target, weights):
    scores = np.matmul(feature, weights)
    return np.sum(target*scores - np.log(1 + np.exp(scores)))


class LogisticRegression():
    def __init__(self, X, y, step=3000, learning_rate=.01, intercept=True):
        self.intercept = intercept
        if self.intercept:
            self.X = np.hstack((np.ones((X.shape[0], 0)), X))
        else:
            self.X = X
        self.y = y
        self.step = step
        self.learning_rate = learning_rate
        
    def fit(self):
        self.w = np.zeros(self.X.shape[1])
        
        for _ in range(self.step):
            scores = np.matmul(self.X, self.w)
            y_pred = np.where(sigmoid(scores) > 0.5,
                              np.ones(self.X.shape[0]),
                              np.zeros(self.X.shape[0]))
            
            exp = np.exp(-np.dot(self.y, y_pred))
            gradient = -exp/(1+exp) * np.matmul(self.X.T, self.y)
            # self.w += self.learning_rate * np.matmul(self.X.T, self.y-y_pred)
            self.w -= self.learning_rate * gradient
        # return self.w
    
    def predict(self, test_data = None, output = "label", cutoff = .5):
        try:
            if test_data is None:
                test_data = self.X
            
            label = sigmoid(np.matmul(test_data, self.w))
            if output == "label":
                return np.where(label > cutoff,
                                np.ones(test_data.shape[0]),
                                np.zeros(test_data.shape[0]))
            elif output == "probability":
                return label
            else:
                print("Check input!")
        
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
    model = LogisticRegression(X, Y_)
    model.fit()
    Y_pred = model.predict()
    print(model.w)
    print(get_accuracy(Y_, Y_pred))
