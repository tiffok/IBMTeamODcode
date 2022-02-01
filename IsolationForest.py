#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 14:28:03 2022

@author: shaneraudenbush
"""

#https://towardsdatascience.com/use-the-isolated-forest-with-pyod-3818eea68f08

import numpy as np
import pandas as pd
from pyod.models.iforest import IForest
df = pd.read_csv('https://raw.githubusercontent.com/DaviRolim/UCI-Wine-DataSet/master/wine.csv', header=None)
X = df.loc[1:,1:]
y = df.loc[1:,0]

#When you do unsupervised learning, it is always a safe step to standardize the predictors
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)
X = pd.DataFrame(X)

from sklearn.decomposition import PCA
pca = PCA(2)
x_pca = pca.fit_transform(X)
x_pca = pd.DataFrame(x_pca)
x_pca.columns=['PC1','PC2']
x_pca.head()

y_color = np.where(y=='Iris-setosa','red',
            np.where(y=='Iris-versicolor','blue','green'))
x_pca.plot(kind='scatter', x='PC1', y='PC2',c=y_color)
pca.show()

clf1 = IForest(behaviour="new", max_samples=100) 
clf1.fit(X)

# clf.decision_function: Predict raw anomaly score of X using the fitted detector.
# We apply the model to the test data X_test to get the outlier scores.
y_scores = clf1.decision_function(X)  # outlier scores
y_scores = pd.Series(y_scores)
y_scores.head()

import matplotlib.pyplot as plt
plt.hist(y_scores, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram for Model clf1 Anomaly Scores")
plt.show()

X_cluster = X.copy()
X_cluster['distance'] = y_scores
X_cluster['cluster'] = np.where(X_cluster['distance']<0.05, 0, 1)
X_cluster['cluster'].value_counts()

X_cluster.groupby('cluster').mean()

clf2 = IForest(behaviour="new", max_samples=80) 
clf2.fit(X)

# clf.decision_function: Predict raw anomaly score of X using the fitted detector.
y_scores = clf2.decision_function(X)  # outlier scores
y_scores = pd.Series(y_scores)
y_scores.head()

plt.hist(y_scores, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram for Model clf2 Anomaly Scores")
plt.show()

X_cluster = X.copy()
X_cluster['distance'] = y_scores
X_cluster['cluster'] = np.where(X_cluster['distance']<0.04, 0, 1)
X_cluster['cluster'].value_counts()

X_cluster.groupby('cluster').mean()

clf3 = IForest(behaviour="new", max_samples=60) 
clf3.fit(X)

# clf.decision_function: Predict raw anomaly score of X using the fitted detector.
y_scores = clf3.decision_function(X)  # outlier scores
y_scores = pd.Series(y_scores)
y_scores.head()

plt.hist(y_scores, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram for Model clf3 Anomaly Scores")
plt.show()

X_cluster = X.copy()
X_cluster['distance'] = y_scores
X_cluster['cluster'] = np.where(X_cluster['distance']<0.03, 0, 1)
X_cluster['cluster'].value_counts()

X_cluster.groupby('cluster').mean()

from pyod.models.combination import aom, moa, average, maximization
from pyod.utils.utility import standardizer

# The predictions of the training data can be obtained by clf.decision_scores_.
# It is already generated during the model building process.
scores = pd.DataFrame({'clf1': clf1.decision_scores_,
                             'clf2': clf2.decision_scores_,
                             'clf3': clf3.decision_scores_})

# Combination by average
y_by_average = average(scores)
             
import matplotlib.pyplot as plt
plt.hist(y_by_average, bins='auto')  # arguments are passed to np.histogram
plt.title("Combination by average")
plt.show()

df_1 = X.copy()
df_1['y_by_average_score'] = y_by_average
df_1['y_by_average_cluster'] = np.where(df_1['y_by_average_score']<2, 0, 1)
df_1['y_by_average_cluster'].value_counts()

df_1.groupby('y_by_average_cluster').mean().round(2)

# Combination by max
y_by_maximization = maximization(scores)
             
import matplotlib.pyplot as plt
plt.hist(y_by_maximization, bins='auto')  # arguments are passed to np.histogram
plt.title("Combination by max")
plt.show()

df_1 = X.copy()
df_1['y_by_maximization_score'] = y_by_maximization
df_1['y_by_maximization_cluster'] = np.where(df_1['y_by_maximization_score']<2, 0, 1)
df_1['y_by_maximization_cluster'].value_counts()

df_1.groupby('y_by_maximization_cluster').mean().round(2)

