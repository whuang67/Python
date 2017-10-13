# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 09:31:13 2017

@author: whuang67
"""

import matplotlib.pyplot as plt
import numpy as np


def Norm_Density(x, mean=0., std=1.):
    return 1./(2*np.pi*std**2)**.5 * np.exp(-(x-mean)**2/(2*std**2))

def visualization(mean=0., std=1., start=None, end=None):
    # Initialize the range if nothing is defined
    if start is None or start <= mean-4*std:
        start = mean-4*std
    if end is None or end >= mean+4*std:
        end = mean+4*std
    
    # Draw the distribution line
    step = 8*std/1000
    x = np.arange(mean-4*std, mean+4*std, step)
    plt.plot(x, Norm_Density(x, mean=mean, std=std))
    
    x_ = np.arange(start, end, step)
    plt.fill_between(x_, Norm_Density(x_, mean, std), color="red")
    plt.show()


# 3.2
visualization(start = -1.13) ### a
visualization(end = 0.18) ### b
visualization(start = 8) ### c
visualization(start = -0.5, end = 0.5) ### d

# 3.10
visualization(mean = 55, std = 6, end = 48) ### a
visualization(mean = 55, std = 6, start = 60, end = 65) ### b

visualization(mean = 55, std = 6, end = 54) ### d

# 3.12
visualization(mean = 72.6, std = 4.78, end = 80) ### a
visualization(mean = 72.6, std = 4.78, start = 60, end = 80) ### b

visualization(mean = 72.6, std = 4.78, start = 70) ### d