# -*- coding: utf-8 -*-
"""
Created on Fri Jul 07 20:07:33 2017

@author: Wenke Huang
@language: Python 2.7

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

# Find the subset of all the rows
def getPool(matrix):
    pool = []
    for row in matrix:
        pool.extend(getSubsets(row))
    return pool

# Find the support of each pattern
def getSupport(matrix):
    itemsets = getPool(matrix)
    sup_dict = {}
    for pattern in itemsets:
        sup_dict[pattern] = sup_dict.get(pattern, 0) + 1
    return sup_dict

# Find the frequent pattern based on min_sup
def getFreqPattern(sup_dict, min_sup):
    freq_pattern = {}
    for pattern, count in sup_dict.items():
        if count >= min_sup:
            freq_pattern[pattern] = count
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
    support = getSupport(matrix)
    freq_pattern = getFreqPattern(support, min_sup)
    supersets = getSupersets(freq_pattern)
    closed_pattern = {}
    for key1, value1 in supersets.items():
        cond = [support[key2] != support[key1] for key2 in value1]
        if all(cond):
            closed_pattern[key1] = support[key1]
    return closed_pattern

# Find the max pattern
def getMaxPattern(matrix, min_sup):
    support = getSupport(matrix)
    freq_pattern = getFreqPattern(support, min_sup)
    supersets = getSupersets(freq_pattern)
    max_pattern = {}
    for key1, value1 in supersets.items():
        cond = [support[key2] < min_sup for key2 in value1]
        if all(cond):
            max_pattern[key1] = support[key1]
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
# freq = getFreqPattern(getSupport(matrix), min_sup)
#for element in ArrangeOutput(freq):
#    sys.stdout.write(element + '\n')
#sys.stdout.write('\n')
#closedFreq = getClosedPattern(matrix, min_sup)
#for element in ArrangeOutput(closedFreq):
#    sys.stdout.write(element + '\n')
#sys.stdout.write('\n')
#maxFreq = getMaxPattern(matrix, min_sup)
#for element in ArrangeOutput(maxFreq):
#    sys.stdout.write(element + '\n')

