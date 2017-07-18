import sys
import math

# Read the data
read = sys.stdin.readlines()
lines = int(read[0].strip())
attributes = read[1].strip().split(",")
train_set = [line.strip().split(",") for line in read[2: (lines+1)]]

        
# Expected Information needed to classify a tuple in D:
def get_infoD(data, target_attr):
    val_freq = {}
    info = 0.0
    for line in data:
        if val_freq.has_key(line[target_attr]):
            val_freq[line[target_attr]] += 1.0
        else:
            val_freq[line[target_attr]] = 1.0
    
    for val in val_freq.keys():
        val_prob = val_freq[val] / len(val_freq.values())
        if val_prob > 0:
            info -= val_prob * math.log(val_prob, 2)
        else:
            info -= 0
    
    return info

# Expected Information needed (after using A to split D into several parts) to classify D:
def get_infoAttr(data, attr, target_attr):
    val_freq = {}
    subset_info = 0.0

    for line in data:
        if (val_freq.has_key(line[attr])):
            val_freq[line[attr]] += 1.0
        else:
            val_freq[line[attr]] = 1.0

    for val in val_freq.keys():
        val_prob = val_freq[val] / sum(val_freq.values())
        data_subset = [line for line in data if line[attr] == val]
        if val_prob > 0:
            subset_info += val_prob * get_infoD(data_subset, target_attr)
        else:
            subset_info += 0

    return subset_info

# Information Gain
def gain(data, attr, target_attr):
    return get_infoD(train_set, target_attr) - \
           get_infoAttr(train_set, attr, target_attr)

# Calculate the SplitInfo:
def get_SplitInfo(data, attr, target_attr):
    val_freq = {}
    SplitInfo = 0.0

    for line in data:
        if (val_freq.has_key(line[attr])):
            val_freq[line[attr]] += 1.0
        else:
            val_freq[line[attr]] = 1.0
    
    for val in val_freq.keys():
        val_prob = val_freq[val] / len(val_freq.values())
        if val_prob > 0:
            SplitInfo -= val_prob * math.log(val_prob, 2)
        else:
            SplitInfo -= 0.0
            
    return SplitInfo

# Gain Ratio
def ratio(data, attr, target_attr):
    return gain(data, attr, target_attr)/get_SplitInfo(data, attr, target_attr)

# Print out the output
maxGain, maxVar_Gain, maxRatio, maxVar_Ratio = -1000, None, -1000, None

for i in range(len(attributes)-1):
    if gain(train_set, i, -1) > maxGain:
        maxGain = gain(train_set, i, -1)
        maxVar_Gain = i
    if ratio(train_set, i, -1) > maxRatio:
        maxRatio = ratio(train_set, i, -1)
        maxVar_Ratio = i

print attributes[maxVar_Gain]
print attributes[maxVar_Ratio]
