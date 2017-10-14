# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:03:06 2017

@author: whuang67
"""

import numpy as np

def distance(x1, x2, n = 2):
    # x1 and x2 is numpy arraies with same length here
    if len(x1) == len(x2):
        return np.sum((x1-x2)**n)**(1/n)
    else:
        print("Invalid input!")

def distance2(x1, x2, s=None):
    # x1 and x2 is numpy arraies with same length here
    if len(x1) == len(x2):
        if s is None:
            s = np.diag(np.ones(len(x1)))
        return np.matmul(np.matmul((x1 - x2).T, s), (x1-x2))**.5
    else:
        print("Invalid input!")

class KNearestNeighbors():
    def __init__(self, X, y, k_neighbors = 5):
        self.X = X
        self.y = y
        self.k_neighbors = k_neighbors
    
    def predict(self, dist = "Euclidean", test_data = None):
        if test_data is None:
            test_data = self.X
        # idx = np.ones(shape = (self.X.shape[0], k_neighbors))
        
        output = np.zeros(test_data.shape[0])
        for j, target in enumerate(test_data):
            idx = np.repeat(-1., self.X.shape[0])
            for i, vector in enumerate(X):
                if dist == "Euclidean":
                    idx[i] = distance(vector, target)
                elif dist == "Mahalanobis":
                    idx[i] = distance2(vector, target)
            
            output[j] = self.y[np.argsort(idx)[:self.k_neighbors]].mean()
        return output


"""
Example!
"""
if __name__ == "__main__":
    ## Prepare dataset
    np.random.seed(1)
    P = 4; N = 200; rho = .5
    V = rho**abs(np.array([[i-j for i in range(1, P+1)] for j in range(1, P+1)]))
    X = np.random.multivariate_normal(mean = np.zeros(P), cov = V, size = N)
    beta = np.array([1, 1, .5, .5])
    Y = np.matmul(X, beta.T) + np.random.normal(size = N)
    
    ## Model fitting
    model = KNearestNeighbors(X, Y) # Default 5-nn
    Y_pred = model.predict(test_data = X)
    print(Y_pred)
    print(model.predict(dist = "Mahalanobis", test_data = X))
