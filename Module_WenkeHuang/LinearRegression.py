# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 14:04:34 2017

@author: whuang67
"""


import numpy as np
import scipy.stats as ss


class Linear_Regression():
    def __init__(self, X, y, intercept = True):
        self.intercept = intercept
        if self.intercept:
            self.X = np.hstack((np.ones((X.shape[0], 1)), X))
        else:
            self.X = X
        self.y = y
        self.w = np.matmul(np.matmul(np.linalg.inv(np.matmul(self.X.T, self.X)), self.X.T), self.y)
        
        ### Summary information
        self.RSS = np.sum((self.y - self.predict())**2)
        self.degrees_of_freedom = self.X.shape[0] - self.X.shape[1]
        
        self.sigma_squared = self.RSS/self.degrees_of_freedom
        self.sigma_betas = np.diag(np.linalg.inv(np.matmul(self.X.T, self.X))*self.sigma_squared)**.5


    def predict(self, test_data = None, interval = None):
        if test_data is None:
            test_data = self.X
        elif self.intercept:
            test_data = np.hstack((np.ones((test_data.shape[0], 1)), test_data))
        
        fit = np.matmul(test_data, self.w)
        if interval is None:
            return fit


    def summary(self):
        
        print("Residual standard error: {:.4f}".format(self.sigma_squared**.5))
        print("Degrees of Freedom: {:d}".format(self.degrees_of_freedom))
        SYY = ((self.y - self.y.mean())**2).sum()
        print("R squared: {:.4f}\n".format(1 - self.RSS/SYY))
        
        t = self.w/self.sigma_betas
        p = 2*ss.t.sf(t, self.degrees_of_freedom)
        
        print("Estimate | Std Err | t stat | p value")
        for a, b, c, d in zip(self.w, self.sigma_betas, t, p):
            print("{:.4f} | {:.4f} | {:.4f} | {:.4f}".format(a, b, c, d))


def ANOVA(model_reduced, model_full = None):
    if model_full is not None:
        n = (model_reduced.RSS - model_full.RSS)/(model_reduced.degrees_of_freedom - model_full.degrees_of_freedom)
        d = model_full.RSS/model_full.degrees_of_freedom
        F = n/d
        return F


def confidence_interval(model, alpha = .05):
    cri_val = ss.t.ppf(1-alpha/2., model.degrees_of_freedom)

    lower_bound = model.w - model.sigma_betas * cri_val
    upper_bound = model.w + model.sigma_betas * cri_val
    print("{}% Confidence Interval".format((1-alpha)*100))
    for x, y in zip(lower_bound, upper_bound):
        print("({:.4f}, {:.4f})".format(x, y))


def dignostics_plot(model):
    import matplotlib.pyplot as plt
    pred = model.predict()
    resid = model.y - pred
    h = np.diag(np.matmul(np.matmul(model.X, np.linalg.inv(np.matmul(model.X.T, model.X))), model.X.T))
    standard_resid = resid/(model.sigma_squared*(1-h))**.5
    
    ## Residuals vs Fitted
    plt.scatter(pred, resid)
    plt.xlabel("Fitted"); plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted")
    plt.show()
    
    ## Normal Q-Q
    mean = standard_resid.mean()
    std = standard_resid.std()
    theoretical_q = ss.norm.ppf(np.arange(1., model.X.shape[0]+1.)/(model.X.shape[0]+1.),
                                loc=mean, scale=std)
    plt.scatter(theoretical_q, sorted(standard_resid))
    plt.xlabel("Theoretical Quantitles"); plt.ylabel("Standard Residuals")
    plt.title("Normal Q-Q")
    plt.show()

    ## Scale-Location
    plt.scatter(pred, abs(standard_resid)**.5)
    plt.xlabel("Fitted"); plt.ylabel("sqrt |Standard Residuals|")
    plt.title("Scale-Location")
    plt.show()
    
    ## Residuals vs Leverage
    plt.scatter(h, standard_resid)
    plt.xlabel("Leverage"); plt.ylabel("Standard Residuals")
    plt.title("Residuals vs Leverage")
    plt.show()



if __name__ == "__main__":
    
    ## Chapter 1
    import pandas as pd
    USmelanoma = pd.read_csv("http://www.unc.edu/%7Enielsen/soci252/activities/USmelanoma.csv")

    X = USmelanoma[["latitude"]].values
    y = USmelanoma["mortality"].values

    model = Linear_Regression(X, y)
    weight = model.w
    print(weight)
    y_pred = model.predict()
    print(y_pred)
    print(y - y_pred)
    print(model.summary())
    
    import matplotlib.pyplot as plt
    plt.scatter(X, y)
    plt.plot(X, weight[0] + X*weight[1])
    plt.show()
    
    
    ## Chapter 2
    X = USmelanoma[["ocean"]].values
    X_ = np.ones(shape = (X.shape[0], 1))
    for i, val in enumerate(X):
        if val == ["yes"]:
            X_[i] = [1.]
        else:
            X_[i] = [0.]

    model = Linear_Regression(X_, y)
    print(model.summary())
    
    
    X_ = np.hstack((USmelanoma[["latitude"]].values, X_))
    model = Linear_Regression(X_, y)
    print(model.summary())
    
    
    ## Chapter 3
    import os
    os.chdir("C:/users/whuang67/downloads")
    savings = pd.read_csv("savings.csv")
    X_full = savings[["pop15", "pop75", "dpi", "ddpi"]]
    X_reduced = savings[["dpi", "ddpi"]].values
    y = savings["sr"].values
    model_full = Linear_Regression(X_full, y)
    print(model_full.summary())
    model_reduced = Linear_Regression(X_reduced, y)
    print(model_reduced.summary())
    print(ANOVA(model_reduced, model_full))
    
    print(confidence_interval(model_full))
    # Not finished
    
    
    ## Chapter 4
    dignostics_plot(model_full)
    
    
    ## Chapter 5