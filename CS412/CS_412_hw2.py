# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 08:39:22 2017

@author: Wenke Huang
"""
### Only Question 1(2) is a coding problem


# Question 1
# 1
# 2
import numpy as np
matrix = np.array([[2,0,0], [0,3,4], [0,4,9]])
## a (Yes)
a = np.array([1,0,0])
print matrix.dot(a)
## b (Yes)
b = np.array([0,1,2])
print matrix.dot(b)
## c (No)
c = np.array([1,1,1])
print matrix.dot(c)

