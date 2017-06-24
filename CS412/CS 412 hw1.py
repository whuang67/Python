# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 12:39:16 2017

@author: whuang67
"""

import pandas as pd
import scipy.stats as ss
import scipy.spatial.distance as ssd
import pylab
from sklearn import preprocessing

file = "C:/users/whuang67/downloads/data/data/scores.txt"
dat = pd.read_csv(file, sep = "\t", header = None, names = ["id", "mid","final"])

## Question 1
## 1(a)
min(dat.mid)
max(dat.mid)

## 1(b)
dat.mid.quantile(q = [.25, 0.5, 0.75])
## 1(c)
a = round(dat.mid.mean(), 2)
b = round(dat.mid.std(), 2)
## 2
ss.probplot(dat.mid, dist = "norm", plot = pylab)
pylab.title("Q-Q plot of midterm score")
pylab.show()
ss.probplot(dat.final, dist = "norm", plot = pylab)
pylab.title("Q-Q plot of final score")
pylab.show()
# dat.final.quantile(q = [.25, .5, .75])


## Question 2
file2 = "C:/users/whuang67/downloads/data/data/inventories.txt"
dat2 = pd.read_csv(file2, sep = "\t", header = 'infer', index_col = 0)

## 1
Jaccard_coefficient = round(107/(107+19+31), 2)
print(Jaccard_coefficient)
## 2(a)
ssd.minkowski(dat2[:1], dat2[1:2], 1)
## 2(b)
ssd.minkowski(dat2[:1], dat2[1:2], 2)
## 2(c)
firstline = dat2[:1].values
secondline = dat2[1:2].values
(abs(firstline-secondline)).max()

## 3
1-ssd.cosine(dat2[0:1], dat2[1:2])

## Question 3
## 1

dat_mid = preprocessing.scale(dat.mid)
round(dat_mid.mean(), 2)
dat_mid.std()

## 2
round((90-dat.mid.mean())/dat.mid.std(), 2)
