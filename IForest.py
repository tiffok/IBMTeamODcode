#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 18:43:40 2022

@author: shaneraudenbush
"""

#https://betterprogramming.pub/anomaly-detection-with-isolation-forest-e41f1f55cc6

import plotly.express as px
from sklearn.ensemble import IsolationForest
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/DaviRolim/UCI-Wine-DataSet/master/wine.csv', header=None)
df.columns = df.iloc[0]
df = df.drop(index = 0)
X = df.iloc[:, 1:]
Y = df.iloc[:, 1]

iforest = IsolationForest(n_estimators=100, max_samples='auto', 
                          contamination='auto', max_features=1, 
                          bootstrap=False, n_jobs=-1, random_state=1)

pred= iforest.fit_predict(X)
df['scores']=iforest.decision_function(X)
df['anomaly_label']=pred

df[df.anomaly_label==-1]

df['anomaly']=df['anomaly_label'].apply(lambda x: 'outlier' if x==-1  else 'inlier')
fig=px.histogram(df,x='scores',color='anomaly')
fig.show()