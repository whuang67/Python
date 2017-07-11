# -*- coding: utf-8 -*-
"""
Created on Wed May 31 09:11:10 2017

@author: whuang67
"""

# from time import time
data = [115., 140., 175.]


def featureScaling(arr):
    b = []
    for stuff in arr:
        a = (stuff - min(arr))/(max(arr)-min(arr))
        b.append(a)
    return b
print(featureScaling(data))

from sklearn.preprocessing import MinMaxScaler
import numpy

weights = numpy.array(data)
scaler = MinMaxScaler()
rescaled_weight = scaler.fit_transform(weights)
print(rescaled_weight)
