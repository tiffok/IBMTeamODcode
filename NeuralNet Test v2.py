#%%

#autoencoder article: https://towardsdatascience.com/anomaly-detection-with-autoencoder-b4cdce4866a6

import scipy.io
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import combo
#from pyod.models.hbos import HBOS
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pyod.models.auto_encoder import AutoEncoder


def actualplot(a_out):
    print("Outlier Counts: 0: inliers 1: outliers")
    print(a_out.value_counts())
    print("plot of actual outliers")
    sns.scatterplot(data = x_pca, x = 'PC1', y = 'PC2', hue = a_out)
    plt.show()

def predplot(y_pred):
    print("plot of predicted outliers")
    sns.scatterplot(data = x_pca, x = 'PC1', y = 'PC2', hue = y_pred)
    plt.show()

def modelEvals(a_out, y_pred):
    #potential confusion matrix
    from sklearn.metrics import confusion_matrix
    y_true = a_out
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


#drop outlier label column
data.drop('Y', inplace=True, axis=1)

#scale data
scaler = StandardScaler()
scaled = scaler.fit_transform(data)
scaled = pd.DataFrame(scaled)

#print(scaled)


#run PCA
pca = PCA(2)
x_pca = pca.fit_transform(scaled)
x_pca = pd.DataFrame(x_pca)
x_pca.columns=['PC1','PC2']

print(x_pca.head())

sns.scatterplot(data=x_pca, x = 'PC1', y = 'PC2')
plt.show()

#neuron model examples
m1 = [25,2,2,25]
m2 = [25,10,2,10,25]
m3 = [25,15,10,2,10,15,25]

#create model 1
clf1 = AutoEncoder(hidden_neurons=m1)

#generate OD scores
clf1.fit(x_pca)
y_scores = clf1.decision_scores_


#outlier label predictions
y_pred = clf1.predict(x_pca) 

# clf.decision_function: Predict raw anomaly score of X using the fitted detector.
y_df = clf1.decision_function(x_pca)

y_pred1 = pd.Series(y_pred)
y_df = pd.Series(y_df)

plt.hist(y_df, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram for Model Clf1 Anomaly Scores")
plt.show()

#value table
df_test = x_pca.copy()
df_test['score'] = y_df
#for lympho
#df_test['cluster'] = np.where(df_test['score']<7, 0, 1)

#for glass
#df_test['cluster'] = np.where(df_test['score']<6, 0, 1)

#for wine
#df_test['cluster'] = np.where(df_test['score']<4.75, 0, 1)

#varying
#df_test['cluster'] = np.where(df_test['score']<6, 0, 1)
#model generated

pred = clf1.fit_predict(x_pca)
df_test['cluster'] = pred

df_test['cluster'].value_counts()

df_test.groupby('cluster').mean()
print("Outlier Counts: 0: inliers 1: outliers")
print(df_test['cluster'].value_counts())

y_pred = df_test['cluster']

predplot(y_pred)
actualplot(a_out)
modelEvals(a_out, y_pred)

#model 2
clf2 = AutoEncoder(hidden_neurons =m2)
clf2.fit(x_pca)


# clf.decision_function: Predict raw anomaly score of X using the fitted detector.
y_scores2 = clf2.decision_function(x_pca)  # outlier scores
y_scores2 = pd.Series(y_scores2)

plt.hist(y_scores2, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram for Model Clf2 Anomaly Scores")
plt.show()

df_test2 = x_pca.copy()
df_test2['score'] = y_scores2
#for lympho
#df_test2['cluster'] = np.where(df_test2['score']<7, 0, 1)

#for glass
#df_test2['cluster'] = np.where(df_test2['score']<6, 0, 1)
#for wine
#df_test2['cluster'] = np.where(df_test2['score']<4.75, 0, 1)

#varying
#df_test2['cluster'] = np.where(df_test2['score']<6, 0, 1)

#model generated
pred = clf2.fit_predict(x_pca)
df_test2['cluster'] = pred

df_test2['cluster'].value_counts()

df_test2.groupby('cluster').mean()
print("Outlier Counts: 0: inliers 1: outliers")
print(df_test2['cluster'].value_counts())

y_pred = df_test2['cluster']

predplot(y_pred)
actualplot(a_out)
modelEvals(a_out,y_pred)

#model 3
clf3 = AutoEncoder(hidden_neurons =m3)
clf3.fit(x_pca)


# clf.decision_function: Predict raw anomaly score of X using the fitted detector.
y_scores3 = clf3.decision_function(x_pca)  # outlier scores
y_scores3 = pd.Series(y_scores3)

plt.hist(y_scores3, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram for Model Clf3 Anomaly Scores")
plt.show()

df_test3 = x_pca.copy()
df_test3['score'] = y_scores2
#for lympho
#df_test3['cluster'] = np.where(df_test3['score']<7, 0, 1)

#for glass
#df_test3['cluster'] = np.where(df_test3['score']<6, 0, 1)
#for wine
df_test3['cluster'] = np.where(df_test3['score']<4.75, 0, 1)

#varying (pima: 2.5, ionosphere: 6)
#df_test3['cluster'] = np.where(df_test3['score']<6, 0, 1)

#model generating outliers
pred = clf3.fit_predict(x_pca)
df_test3['cluster'] = pred


df_test3['cluster'].value_counts()

df_test3.groupby('cluster').mean()
print("Outlier Counts: 0: inliers 1: outliers")
print(df_test3['cluster'].value_counts())

#predictions by threshold
y_pred = df_test3['cluster']

predplot(y_pred)
actualplot(a_out)
modelEvals(a_out, y_pred)

from pyod.models.combination import aom, moa, average, maximization

# Put all the predictions in a data frame
train_scores = pd.DataFrame({'clf1': clf1.decision_scores_,
                             'clf2': clf2.decision_scores_,
                             'clf3': clf3.decision_scores_
                            })



test_scores  = pd.DataFrame({'clf1': clf1.decision_function(x_pca),
                             'clf2': clf2.decision_function(x_pca),
                             'clf3': clf3.decision_function(x_pca) 
                            })


#average aggregation method

#standardize scores
from pyod.utils.utility import standardizer
train_scores_norm, test_scores_norm = standardizer(train_scores,test_scores)

# Combination by average
y_by_average = average(test_scores_norm)
             
import matplotlib.pyplot as plt
plt.hist(y_by_average, bins='auto')  # arguments are passed to np.histogram
plt.title("Combination by average")
plt.show()

#get outlier predictions

df_test = pd.DataFrame(x_pca)
df_test['y_by_average_score'] = y_by_average
#for lympho
#df_test['y_by_average_cluster'] = np.where(df_test['y_by_average_score']<2, 0, 1)

#for glass
#df_test['y_by_average_cluster'] = np.where(df_test['y_by_average_score']<2, 0, 1)

#for wine
df_test['y_by_average_cluster'] = np.where(df_test['y_by_average_score']<2, 0, 1)

#varying (pima: 0.5, ionosphere: 0.25)
#df_test['y_by_average_cluster'] = np.where(df_test['y_by_average_score']<0.25, 0, 1)
print("Outlier Counts: 0: inliers 1: outliers")
print(df_test['y_by_average_cluster'].value_counts())

y_pred = df_test['y_by_average_cluster']

#summary statistics
df_test.groupby('y_by_average_cluster').mean()

#plot
predplot(y_pred)
actualplot(a_out)
modelEvals(a_out, y_pred)




#%%
'''
from sklearn.metrics import auc, roc_curve
#for model 1
tpr1, fpr1, threshold1 = roc_curve(a_out,y_df) 
roc_auc = auc(fpr1,tpr1)


plt.title('Receiver Operating Characteristic')
plt.plot(fpr1, tpr1, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
'''
# %%
