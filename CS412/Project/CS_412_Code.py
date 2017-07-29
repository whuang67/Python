"""
The code of CS 412 Project
Analysis of Knowledge Discover in Databases Papers on Data Mining Models
author: Wenke Huang
"""

## Useful Packages ###########################################################
import os
import re
import itertools
import scipy.stats as ss
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, \
                            classification_report, f1_score, silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

## Load Papers and Preprocess ################################################
os.chdir("C:/Users/whuang67/downloads/txt/txt/kdd15")
dat =[]
for i in range(0, 100):
    if i <= 37:
        path = 'kdd15-p' + str(10*i+9)+'.txt'
    elif i <= 62:
        path = 'kdd15-p' + str(10*i+7)+'.txt'
    elif i <= 78:
        path = 'kdd15-p' + str(10*i+5)+'.txt'
    elif i <= 90:
        path = 'kdd15-p' + str(10*i+15)+'.txt'
    else:
        path = 'kdd15-p' + str(10*i+25)+'.txt'
    document = open(path)
    data = document.read()
    dat.append(data)

os.chdir("C:/Users/whuang67/downloads")
document = open('aaa2.txt')
data = document.readlines()
dat_cleaned = [x for x in data if x.strip()]
paper = {}
for line in dat_cleaned:
    if line[0:7] != 'SESSION' and line [0:5] != 'ABCDE':
        paper_subj = line.strip()
        paper[paper_subj] = []
    if line[0:5] == 'ABCDE' and paper_subj != None:
        paper[paper_subj].append(line[6:].strip())

papers = []
for line in dat_cleaned:
    if line[0:7] != 'SESSION' and line [0:5] != 'ABCDE':
        papers.append(line)
        
authors = {}
for co_authors in paper.values():
    for author in co_authors:
        authors[author] = authors.get(author, 0) + 1

# authors['Eric P. Xing'] += authors['Eric Xing']
# del authors['Eric Xing']
# authors['Charalampos Tsourakakis'] += authors['Charalampos E. Tsourakakis']
# del authors['Charalampos E. Tsourakakis']
# authors['Alexander J. Smola'] += authors['Alex J. Smola']
# del authors['Alex J. Smola']
# authors['Charu Aggarwal'] += authors['Charu C. Aggarwal']
# del authors['Charu C. Aggarwal']

os.chdir("C:/Users/whuang67/downloads")
document = open('Category.txt')
data = document.readlines()
response = [int(e.strip()) for e in data]


## Data Transformation #######################################################
dat1 = []
for line in dat:
    dat1.append(re.sub("[^A-Za-z .,|\n]+", "", line.lower()))

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dat1)
svd = TruncatedSVD(n_components = 100)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(X)

## Frequent Pattern Mining ###################################################
iceburg = {}
for name, item in authors.items():
    if item >= 2:
        iceburg[name] = authors[name]

co_workers = {}
author_pairs = list(itertools.combinations(iceburg.keys(), 2))
for authors_ in paper.values():
    for a in author_pairs:
        if a[0] in authors_ and a[1] in authors_:
            count = co_workers.get(a, [0, 0])[0]
            co_workers[a] = [count + 1,
                             (count + 1.0)/iceburg[a[0]]]


co_w_sub = {key: value for key, value in co_workers.items() \
            if value[1] >= 0.5 and value[0] >= 2}

freq_pat_m = {}
for key, value in co_w_sub.items():
    s_A = authors[key[0]]/100.0
    s_B = authors[key[1]]/100.0
    s_AB = value[0]/100.0
    table = np.array([[s_AB, s_A-s_AB], [s_B-s_AB, 1-s_A-s_B+s_AB]]*100)
    freq_pat_m[key] = [
            s_AB / (s_A*s_B),
            ss.chi2_contingency(table)[0],
            s_AB / (s_A*s_B)**.5,
            (s_AB / s_A + s_AB / s_B)*.5]

## Classification (Naive Bayes) ##############################################

X_train, X_test, y_train, y_test = train_test_split(
        X, response, test_size = 20, random_state = 6) ### 6

Classifier = BernoulliNB()
Classifier.fit(X_train, y_train)
pred = Classifier.predict(X_train)
prob = pd.DataFrame(Classifier.predict_proba(X_train),
                    columns = ['0', '1'])


fpr, tpr, _ = roc_curve(y_train, prob['1'])
auc = roc_auc_score(y_train, prob['1'])
plt.plot(fpr, tpr)
plt.title('Figure 1: ROC curve of training set')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.text(0.6, 0.4, 'AUC = '+str(round(auc, 4)),
         fontsize = 15)
plt.show()

print confusion_matrix(y_train, pred)
print classification_report(y_train, pred, digits = 4)

pred_test = Classifier.predict(X_test)

prob_test = pd.DataFrame(Classifier.predict_proba(X_test),
                         columns = ['0', '1'])
fpr, tpr, _ = roc_curve(y_test, prob_test['1'])
auc = roc_auc_score(y_test, prob_test['1'])
plt.plot(fpr, tpr)
plt.title('Figure 2: ROC curve of testing set')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.text(0.6, 0.4, 'AUC = '+str(round(auc, 4)),
         fontsize = 15)
plt.show()
print confusion_matrix(y_test, pred_test)
print classification_report(y_test, pred_test, digits = 4)

## K-Means Clustering ########################################################

for i in range(2, 8):
    Clst = KMeans(n_clusters = i,
                  random_state = 7)
    Clst.fit(X)
    Cluster = Clst.predict(X)
    print(i, round(silhouette_score(X, Cluster), 4))

Clst = KMeans(n_clusters = 2,
              random_state = 7)
Clst.fit(X)
Cluster = Clst.predict(X)

## DBSCAN ####################################################################

Clst = DBSCAN(eps = 1) ## 0.85, 0.9, 0.95, 1, 1.05
Clst.fit(X)
Cluster = Clst.labels_



