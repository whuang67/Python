# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 19:22:21 2017

@author: whuang67
"""

### Change the current working directory if necessary
# import os
# os.chdir("~/")


import numpy as np
import pandas as pd

### Read the dataset into Python
ClaimLevel = pd.read_csv("ClaimLevel.csv")

### Split variable ClaimDate
ClaimLevel[["Year", "Month", "Day"]] = pd.DataFrame(
        ClaimLevel["ClaimDate"].str.split("-").tolist())


### Invalid records
idx = (ClaimLevel["ClaimedAmount"] < ClaimLevel["PaidAmount"]).values
print(ClaimLevel[idx])
ClaimLevel.drop([4104, 12657, 47880, 54902, 83558, 117353,
                 130137, 140342, 142718],
                inplace = True)

### Sort dataset which helps us convert it later
dat1 = ClaimLevel.sort_values(by = ["PolicyId", "Year", "Month"])
dat1.reset_index(drop = True, inplace = True)


### Find out unique Policy ID
unique_customer_num = len(set(dat1["PolicyId"].tolist()))

### Convert the original dataset into a monthly-based one
dat = np.empty(shape = (unique_customer_num, 25))
old_id, count = None, -1
for i, line in enumerate(dat1.as_matrix()):
    current_id, _, current_claim, current_paid, _, current_month, _ = line
    month_idx = 2*int(current_month)

        
    if old_id is None or old_id != current_id:
        count += 1
        dat[count, 0] = current_id
        dat[count, month_idx-1] = current_claim 
        dat[count, month_idx] = current_paid        
        old_id = current_id
        
        
    elif old_id == current_id:
        if dat[count, month_idx-1] != 0 or dat[count, month_idx] != 0:
            dat[count, month_idx-1] += current_claim
            dat[count, month_idx] += current_paid
        else:
            dat[count, month_idx-1] = current_claim
            dat[count, month_idx] = current_paid
        old_id = current_id
        

### Visualize the historical records of customers
import matplotlib.pyplot as plt
def Customer_plot(PolicyId, claim = False, paid = False, diff = True):
    plot = False
    for j in range(len(dat)):
        if dat[j, 0] == PolicyId:
            plot = True
            break
    
    if plot is True:
        claim_ = np.array([dat[j, i] for i in range(len(dat[0])) \
                           if i % 2 == 1])
        paid_ = np.array([dat[j, i] for i in range(1, len(dat[0])) \
                          if i % 2 == 0])
        diff_ = claim_ - paid_
    
        if claim is True:
            plt.plot(claim_, color = "green", label = "Claim Amount")
        
        if paid is True:
            plt.plot(paid_, color = "red", label = "Paid Amount")
        
        if diff is True:
            plt.plot(diff_, color = "orange", label = "Different Amount")
    
        plt.title("Records of Customer " + str(PolicyId))
        plt.xticks(range(12), range(1, 13)); plt.xlabel("Month")
        plt.ylabel("Amount"); plt.legend()
    else:
        print("There is not Policy ID {}! Please check!".format(PolicyId))

plt.figure(figsize = (14, 9.5))
plt.subplot(221)
Customer_plot(10, claim=True, paid=True)
plt.subplot(222)
Customer_plot(591, claim=True, paid=True)
plt.subplot(223)
Customer_plot(958, claim=True, paid=True)
plt.subplot(224)
Customer_plot(1657, claim=True, paid=True)
plt.show()


### Previous 11 months records are predictors
X = dat[:, 1:23]
### Current paid amount is response
y = dat[:, 24]

    
### Perform Z-score transformation
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)


### Split the dataset into training and testing subsets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = .2, random_state = 1)


### Linear Regression
from sklearn.linear_model import LinearRegression
reg_lr = LinearRegression().fit(X_train, y_train)
predict_lr_train = reg_lr.predict(X_train)
predict_lr = reg_lr.predict(X_test)

print("Training error of Linear Regression:\n{}".format(
      mean_squared_error(y_train, predict_lr_train)**.5))
print("Testing error of Linear Regression:\n{}".format(
      mean_squared_error(y_test, predict_lr)**.5))



### Decision Tree
from sklearn.tree import DecisionTreeRegressor
reg_tree = DecisionTreeRegressor(random_state=1).fit(X_train, y_train)

predict_tree_train = reg_tree.predict(X_train)
predict_tree = reg_tree.predict(X_test)

print("Training error of Decision Tree:\n{}".format(
      mean_squared_error(y_train, predict_tree_train)**.5))
print("Testing error of Decision Tree:\n{}".format(
      mean_squared_error(y_test, predict_tree)**.5))



### Random Forest
from sklearn.ensemble import RandomForestRegressor
parameter = {"n_estimators": range(200, 350, 50),
             "max_features": range(3, 22//3)}

reg_rf = GridSearchCV(RandomForestRegressor(random_state = 1,
                                            verbose = 1),
                      parameter,
                      scoring = "neg_mean_squared_error",
                      cv = 5,
                      verbose = 1)

reg_rf.fit(X_train, y_train)
reg_rf_best = reg_rf.best_estimator_

predict_rf_train = reg_rf_best.predict(X_train)
predict_rf = reg_rf_best.predict(X_test)


print("Training error of Random Forest:\n{}".format(
      mean_squared_error(y_train, predict_rf_train)**.5))
print("Testing error of Random Forest:\n{}".format(
      mean_squared_error(y_test, predict_rf)**.5))



### K Nearest Neighbors
from sklearn.neighbors import KNeighborsRegressor
parameter = {"n_neighbors": range(111, 151, 10)}

reg_knn = GridSearchCV(KNeighborsRegressor(),
                   parameter,
                   scoring = "neg_mean_squared_error",
                   cv = 5,
                   verbose = 1)

reg_knn.fit(X_train, y_train)
reg_knn_best = reg_knn.best_estimator_

predict_knn_train = reg_knn_best.predict(X_train)
predict_knn = reg_knn_best.predict(X_test)

print("Training error of KNN:\n{}".format(
      mean_squared_error(y_train, predict_knn_train)**.5))
print("Testing error of KNN:\n{}".format(
      mean_squared_error(y_test, predict_knn)**.5))



### Multilayer Perceptron Neural Network
from keras.models import Sequential
from keras.layers import Dense  #, Dropout

np.random.seed(1)
model = Sequential()
model.add(Dense(22, input_dim = 22, activation = "relu"))
model.add(Dense(11, activation = "relu"))
# model.add(Dropout(.2))
model.add(Dense(5, activation = "relu"))
# model.add(Dropout(.2))
model.add(Dense(1))
model.compile(loss = "mean_squared_error", optimizer = "adam")
model.summary()

np.random.seed(1)
model.fit(X_train, y_train, epochs = 20, verbose = 2)

predict_mlp_train = np.transpose(model.predict(X_train))[0]
predict_mlp = np.transpose(model.predict(X_test))[0]

print("Training error of MLP:\n{}".format(
      mean_squared_error(y_train, predict_mlp_train)**.5))
print("Testing error of MLP:\n{}".format(
      mean_squared_error(y_test, predict_mlp)**.5))

### Ensemble
### Model 1
model1_train = (predict_lr_train + predict_tree_train + predict_rf_train + \
                predict_knn_train + predict_mlp_train)/5
model1 = (predict_lr + predict_tree + predict_rf + \
          predict_knn + predict_mlp)/5
print("Training error of Model 1:\n{}".format(
      mean_squared_error(y_train, model1_train)**.5))
print("Testing error of Model 1:\n{}".format(
      mean_squared_error(y_test, model1)**.5))


### Model 2
model2_train = (predict_lr_train + predict_rf_train + predict_knn_train + \
                predict_mlp_train)/4
model2 = (predict_lr + predict_rf + predict_knn + predict_mlp)/4
print("Training error of Model 2:\n{}".format(
      mean_squared_error(y_train, model2_train)**.5))
print("Testing error of Model 2:\n{}".format(
      mean_squared_error(y_test, model2)**.5))


### Model 3
model3_train = predict_lr_train*.2 + predict_rf_train*.4 + \
               predict_knn_train*.2 + predict_mlp_train*.2
model3 = predict_lr*.2 + predict_rf*.4 + predict_knn*.2 + predict_mlp*.2
print("Training error of Model 3:\n{}".format(
      mean_squared_error(y_train, model3_train)**.5))
print("Testing error of Model 3:\n{}".format(
      mean_squared_error(y_test, model3)**.5))


### Model 4
model4_train = predict_lr_train*.1 + predict_rf_train*.4 + \
               predict_knn_train*.3 + predict_mlp_train*.2
model4 = predict_lr*.1 + predict_rf*.4 + predict_knn*.3 + predict_mlp*.2
print("Training error of Model 4:\n{}".format(
      mean_squared_error(y_train, model4_train)**.5))
print("Testing error of Model 4:\n{}".format(
      mean_squared_error(y_test, model4)**.5))


### Make Prediction
X_ = dat[:, 3:25]
X_ = StandardScaler().fit_transform(X_)
LR = reg_lr.predict(X_)
RF = reg_rf_best.predict(X_)
KNN = reg_knn_best.predict(X_)
MLP = np.transpose(model.predict(X_))[0]
ensemble_model4 = LR*.1 + RF*.4 + KNN*.3 + MLP*.2

### Write output
with open("output.txt", "w") as output:
    output.write("PolicyId,PaidAmount\n")
    for i in range(unique_customer_num):
        output.write(str(int(dat[i, 0]))+","+str(ensemble_model4[i])+"\n")