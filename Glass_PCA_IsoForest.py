#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load packages
import plotly.express as px
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np

# import and clean data
import scipy.io
import pandas as pd

#read in mat file
udata = scipy.io.loadmat("/Users/allisonrudolph/Downloads/glass.mat")


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
X = data

#PCA 
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



# create isolation forest object
iforest = IsolationForest(n_estimators=100, max_samples='auto', 
                          contamination=.04, max_features=1, # contamination of .04 because ODDS website tells us 4% are outliers
                          bootstrap=False, n_jobs=-1, random_state=1)

# predict anomalies
pred= iforest.fit_predict(x_pca)
data['scores']=iforest.decision_function(x_pca)
data['anomaly_label']=pred

data[data.anomaly_label==-1]

#create histogram to show anomalies
data['anomaly']=data['anomaly_label'].apply(lambda x: 'outlier' if x==-1  else 'inlier')
fig=px.histogram(data,x='scores',color='anomaly')
fig.show()
y_pred = np.where(data["anomaly_label"]==-1, 1, 0)


# In[2]:


#confusion matrix with rates
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
y_true = a_out #actual outliers
#need to have a column of the predicted outliers for the confusion matrix (saved as y_pred)
c_mat = confusion_matrix(y_true, y_pred)
tn = c_mat[0,0]
fp = c_mat[0,1]
fn = c_mat[1,0]
tp = c_mat[1,1]

# print in matrix form
print("confusion matrix: ")
print(c_mat)
print("\n")

actno = tn + fp
actyes = tp + fn
#rates
#misclass = (fp + fn) / (actno + actyes)
tpr = tp / actyes
fpr = fp / actno
#tnr = tn/ actno
precision = tp / (fp + tp)
accuracy = (tp + tn) / (actno + actyes)
prevalence = actyes / (actno + actyes)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(a_out,y_pred)

#print results
print("There are {} True Negatives.".format(tn))
print("There are {} False Positives.".format(fp))
print("There are {} False Negatives.".format(fn))
print("There are {} True Positives.".format(tp))
print("Accuracy = {:.2f}".format(accuracy))
print("Precision = {:.2f}".format(precision))
print("Prevalence = {:.2f}".format(prevalence))
print("True Positive Rate = {:.2f}".format(tpr))
#print("False Positive Rate = " + str(fpr))
print("F1 Score = {:.2f}".format(f1) )
print("AUC = {:.2f}".format(auc))
print("\n")

#plot ROC curve
nfpr, ntpr, threshold = roc_curve(a_out, y_pred)
plt.plot(nfpr, ntpr, color = "darkorange", label = "Model")
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()

