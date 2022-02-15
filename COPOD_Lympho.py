#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 13:12:06 2022

@author: shaneraudenbush
"""

import scipy.io
import pandas as pd

#read in mat file
udata = scipy.io.loadmat("lympho.mat")

# create dataframe (add array for column names separately)
data_val = udata['X']

data = pd.DataFrame(data_val)

#actual outlier labels
actuals = pd.DataFrame(udata['y'])

#total data frame (including labeled outliers)

data['Y'] = actuals

#outlier column
a_out = data['Y']

print(data.head())

#drop outlier label column
data.drop('Y', inplace=True, axis=1)

print(data.head())

#COPOD Method
from pyod.models.copod import COPOD

X = data

clf = COPOD()
clf.fit(X)

y_scores = clf.decision_scores_

#Add outlier scores to dataframe
data['outlier scores'] = y_scores

#Label predicted outliers in new column based on outlier score of more than 19
data['predicted outliers'] = (data['outlier scores'] > 19).astype(int)

#confusion matrix with rates
from sklearn.metrics import confusion_matrix
y_true = a_out #actual outliers
#need to have a column of the predicted outliers for the confusion matrix (saved as y_pred)
y_pred = data['predicted outliers']

c_mat = confusion_matrix(y_true, y_pred)
tn = c_mat[0,0]
fp = c_mat[0,1]
fn = c_mat[1,0]
tp = c_mat[1,1]

actno = tn + fp
actyes = tp + fn
#rates
misclass = (fp + fn) / (actno + actyes)
tpr = tp / actyes
fpr = fp / actno
tnr = tn/ actno
precision = tp / (fp + tp)
accuracy = (tp + tn) / (actno + actyes)
prevalence = actyes / (actno + actyes)

#print results
print("There are " + str(tn) + " True Negatives.")
print("There are " + str(fp) + " False Positives.")
print("There are " + str(fn) + " False Negatives.")
print("There are " + str(tp) + " True Positives.")
print("Accuracy = " + str(accuracy))
print("Misclassification rate = " + str(misclass))
print("Precision = " + str(precision))
print("Prevalence = " + str(prevalence))
print("True Positive Rate = " + str(tpr))
print("False Positive Rate = " + str(fpr))
print("True Negative Rate = " + str(tnr))
#print("F1 Score = " + str(f1))