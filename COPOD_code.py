#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 18:31:18 2022

@author: shaneraudenbush
"""

#https://pyod.readthedocs.io/en/latest/

from pyod.models.copod import COPOD
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/DaviRolim/UCI-Wine-DataSet/master/wine.csv', header=None)
df.columns = df.iloc[0]
df = df.drop(index = 0)
X = df.iloc[:, 1:]
Y = df.iloc[:, 1]

clf = COPOD()
clf.fit(X)

y_scores = clf.decision_scores_

df['outlier scores'] = y_scores