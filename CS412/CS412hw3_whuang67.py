# -*- coding: utf-8 -*-
"""
Created on Sat Jul 08 13:44:02 2017

@author: whuang67
"""

import sys
import itertools

# Read the data
read = sys.stdin.readlines()
min_sup = int(read[0].strip())
matrix = [line.strip().split() for line in read[1:]]

# Find subset of each row
def getSubsets(row):
    row = sorted(row)
    subsets = []
    for i in range(1, len(row)+1):
        temp = list(itertools.combinations(row, i))
        subsets.extend(temp)
    return subsets

# Find the largest k for the k-itemset
def LargestK(matrix):
    Length = []
    for row in matrix:
        Length.append(len(row))
    return max(Length)

# Perform Apriori to find frequent pattern
def Apriori(matrix, min_sup):
    k = 1
    freq_pattern = {}
    Largest_K = LargestK(matrix)
    afterPrune = {}
    while k <= Largest_K:
        pool = []
        beforePrune = {}
        for row in matrix:
            items_in_row = getSubsets(row)
            for item in items_in_row:
                if len(item) == k and (elem < afterPrune.keys() for elem in item):
                    pool.append(item)
        for pattern in pool:
            beforePrune[pattern] = beforePrune.get(pattern, 0)+1
        afterPrune = {}
        for pattern, count in beforePrune.items():
            if count >= min_sup:
                afterPrune[pattern] = count
        
        freq_pattern.update(afterPrune)
        k += 1

    return freq_pattern

# Find the supersets
def getSupersets(freq_pattern):
    supersets = {}
    for key1 in freq_pattern.keys():
        superset = []
        for key2 in freq_pattern.keys():
            if set(list(key1)).issubset(set(list(key2))) and key1 != key2:
                superset.append(key2)
        supersets[key1] = superset
    return supersets

# Find the closed pattern
def getClosedPattern(matrix, min_sup):
    freq_pattern = Apriori(matrix, min_sup)
    supersets = getSupersets(freq_pattern)
    closed_pattern = {}
    for key1, value1 in supersets.items():
        cond = [freq_pattern[key2] != freq_pattern[key1] for key2 in value1]
        if all(cond):
            closed_pattern[key1] = freq_pattern[key1]
    return closed_pattern

# Find the max pattern
def getMaxPattern(matrix, min_sup):
    closed_pattern = Apriori(matrix, min_sup)
    supersets = getSupersets(closed_pattern)
    max_pattern = {}
    for key1, value1 in supersets.items():
        cond = [closed_pattern[key2] < min_sup for key2 in value1]
        if all(cond):
            max_pattern[key1] = closed_pattern[key1]
    return max_pattern

# Output following the instruction
def ArrangeOutput(pattern):
    inversed_dict = {}
    for key, value in pattern.items():
        if value in inversed_dict.keys():
            inversed_dict[value].append(" ".join(list(key)))
        else:
            inversed_dict[value] = [" ".join(list(key))]
    
    desired_list = []
    for key in sorted(inversed_dict.keys(), reverse = True):
        temp = sorted(inversed_dict[key])
        element = ["{} [{}]".format(key, p) for p in temp]
        desired_list.extend(element)
    
    return desired_list

# Output
freq = Apriori(matrix, min_sup)
for element in ArrangeOutput(freq):
    sys.stdout.write(element + '\n')
sys.stdout.write('\n')
closed = getClosedPattern(matrix, min_sup)
for element in ArrangeOutput(closed):
    sys.stdout.write(element + '\n')
sys.stdout.write('\n')
maxpattern = getMaxPattern(matrix, min_sup)
for element in ArrangeOutput(maxpattern):
    sys.stdout.write(element + '\n')