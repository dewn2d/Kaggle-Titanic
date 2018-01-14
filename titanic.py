#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 00:31:33 2018

@author: dewn2d
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

X_train = data_train.iloc[:, [2,4,11] ].values
X_test = data_test.iloc[:, [1,3,10]].values
y_train = data_train.iloc[:, 1].values

#plt.scatter( X_train[:,2], y_train, color = 'blue')
#plt.show()

"""
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy = 'mean', axis = 0 )

r, c = np.shape(X_train)
row1 = []
row2 = []
for x in range(0,r):
    if y_train[x] == 1:
            row1.append(X_train[x,:])
    else:
            row2.append(X_train[x,:])

row12 = np.asarray(row1)
row22 = np.asarray(row2)
imputer = imputer.fit(row1[:, 2:3])
row1[:, 2:3] = imputer.transform(row1[:, 2:3])
imputer = imputer.fit(row2[:, 2:3])
row2[:, 2:3] = imputer.transform(row2[:, 2:3])

np.ndarray.mean(col1, axis = 0)
np.ndarray.mean(col2, axis = 0)


sib, sib2 = 0, 0
r, c = np.shape(X_train)
for x in range(0,r):
    if y_train[x] == 1:
        if X_train[x,2] != 0:
            sib += 1
    else:
        if X_train[x,2] != 0:
            sib2 += 1

r, c = np.shape(X_train)
S, C, Q = 0, 0, 0
for x in range(0,r):
    here = X_train[x, 2]
    if y_train[x] == 1: 
        if here == 'C':
            C += 1
        elif here == 'Q':
            Q += 1
        elif here == 'S':
            S += 1
Sa = S/(C+S+Q)
Qa = C/(C+S+Q)
Ca = Q/(C+S+Q)

r, c = np.shape(X_train)
m, w = 0, 0
for x in range(0,r):
    if y_train[x] == 1: 
        if X_train[x, 1] == 'male':
            m += 1
        elif X_train[x, 1] == 'female':
            w += 1
p_women = w/(w+m)

r, c = np.shape(X_train)
c1, c2, c3, C = 0,0,0,0
for x in range(0,r):
    if y_train[x] == 1: 
        if X_train[x, 0] == 1:
            c1 += 1
        elif X_train[x, 0] == 2:
            c2 += 1
        else:
            c3 += 1
        C+=1
C1 = c1/C
C2 = c2/C
C3 = c3/C
"""

from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder

def most_frequent_char(arr, colnum):
    """
    most_frequent_char(arr, colnum)
    takes a column of an np.ndarray with chars and fills the nan
    with the most frequent char in the column

    arr: an np.ndarray
    colnum: the column with chars to be encoded
    """
    #make dummy variables, count and sort descending:
    most_common = pd.get_dummies(arr[:,colnum]).sum().sort_values(ascending=False).index[0] 
    rows, col = arr.shape
    for i in range(0, rows) :
        if pd.isnull(arr[i,colnum]):
            arr[i,colnum] = most_common
    return arr

X_train = most_frequent_char(X_train, 2)
"""
imputer = Imputer(missing_values="NaN", strategy = 'mean', axis = 0 )

imputer = imputer.fit(X_train[:, [2,3]])
X_train[:,[2,3]] = imputer.transform(X_train[:,[2,3]])
imputer = imputer.fit(X_test[:,[2,3]])
X_test[:,[2,3]] = imputer.transform(X_test[:,[2,3]])
"""
le = LabelEncoder()
X_train[:, 1] = le.fit_transform(X_train[:, 1])
X_test[:, 1] = le.fit_transform(X_test[:, 1])

X_train[:, 2] = le.fit_transform(X_train[:, 2])
X_test[:, 2] = le.fit_transform(X_test[:, 2])

# flip the order of class so the higher number means greater class 3 > 1
def flip_class(this, c):
    r, nop = np.shape(this)
    for x in range(0, r):
        if this[x, c] == 1:
            this[x, c] = 3
        elif this[x, c] == 3:
            this[x, c] = 1
    return this

X_train = flip_class(X_train, 0)
X_test = flip_class(X_test, 0)

onehotencoder = OneHotEncoder( categorical_features=[2] )
X_train = onehotencoder.fit_transform(X_train).toarray()
X_test = onehotencoder.fit_transform(X_test).toarray()

X_train = np.delete(X_train, [0], 1)
X_test = np.delete(X_test, [0], 1)

"""
 first 2 rows in set are the classes
 next 2 is the embarkedport
 then gender
 age
 sib
 parch
 ticket price
"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""
# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 35, gamma = 0.1)
classifier.fit(X_train, y_train)
"""

from sklearn.decomposition import PCA
pca = PCA( n_components = None )
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)
explained_variance = pca.explained_variance_ratio_

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(X_train, y_train)

# Predicting the Test set result
y_pred = classifier.predict(X_test)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

#compare current model with previous one
count = 0
prev_out = pd.read_csv('output.csv')
y_prev = prev_out.iloc[:, 1].values
for x in range(0,418):
    if y_prev[x] == y_pred[x]:
        count += 1
        
y_pred=np.matrix(y_pred)
y_pred = y_pred.T

output = np.arange(892,1310).reshape(418,1)
output = np.concatenate((output, y_pred), axis = 1)

df = pd.DataFrame(output,columns = ["PassengerId","Survived"])
df.to_csv("output.csv", index = False )