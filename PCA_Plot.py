#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 17:48:25 2022

@author: shaneraudenbush
"""

#https://towardsdatascience.com/use-the-isolated-forest-with-pyod-3818eea68f08

import numpy as np
import pandas as pd
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

x_pca.plot(kind='scatter', x='PC1', y='PC2',c='green')
