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

def actualplot():
    print("plot of actual outliers")
    sns.scatterplot(data = x_pca, x = 'PC1', y = 'PC2', hue = a_out)
    plt.show()

def predplot(y_pred):
    print("plot of predicted outliers")
    sns.scatterplot(data = x_pca, x = 'PC1', y = 'PC2', hue = y_pred)
    plt.show()


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
clf1.fit(scaled)
y_scores = clf1.decision_scores_


#outlier label predictions
y_pred = clf1.predict(scaled) 

# clf.decision_function: Predict raw anomaly score of X using the fitted detector.
y_df = clf1.decision_function(scaled)

y_pred1 = pd.Series(y_pred)
y_df = pd.Series(y_df)

plt.hist(y_df, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram for Model Clf1 Anomaly Scores")
plt.show()

#value table
df_test = scaled.copy()
df_test['score'] = y_df
#for lympho
df_test['cluster'] = np.where(df_test['score']<7, 0, 1)

#for glass
#df_test['cluster'] = np.where(df_test['score']<6, 0, 1)

#for wine
#df_test['cluster'] = np.where(df_test['score']<4.75, 0, 1)

#varying
#df_test['cluster'] = np.where(df_test['score']<3, 0, 1)

df_test['cluster'].value_counts()

df_test.groupby('cluster').mean()
print(df_test['cluster'].value_counts())

y_pred = df_test['cluster']

predplot(y_pred)
actualplot()

#model 2
clf2 = AutoEncoder(hidden_neurons =m2)
clf2.fit(scaled)


# clf.decision_function: Predict raw anomaly score of X using the fitted detector.
y_scores2 = clf2.decision_function(scaled)  # outlier scores
y_scores2 = pd.Series(y_scores2)

plt.hist(y_scores2, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram for Model Clf2 Anomaly Scores")
plt.show()

df_test2 = scaled.copy()
df_test2['score'] = y_scores2
#for lympho
df_test2['cluster'] = np.where(df_test2['score']<7, 0, 1)

#for glass
#df_test2['cluster'] = np.where(df_test2['score']<6, 0, 1)
#for wine
#df_test2['cluster'] = np.where(df_test2['score']<4.75, 0, 1)

#varying
#df_test2['cluster'] = np.where(df_test2['score']<3, 0, 1)

df_test2['cluster'].value_counts()

df_test2.groupby('cluster').mean()
print(df_test2['cluster'].value_counts())

y_pred = df_test2['cluster']

predplot(y_pred)
actualplot()

#model 3
clf3 = AutoEncoder(hidden_neurons =m3)
clf3.fit(scaled)


# clf.decision_function: Predict raw anomaly score of X using the fitted detector.
y_scores3 = clf3.decision_function(scaled)  # outlier scores
y_scores3 = pd.Series(y_scores3)

plt.hist(y_scores3, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram for Model Clf3 Anomaly Scores")
plt.show()

df_test3 = scaled.copy()
df_test3['score'] = y_scores2
#for lympho
df_test3['cluster'] = np.where(df_test3['score']<7, 0, 1)

#for glass
#df_test3['cluster'] = np.where(df_test3['score']<6, 0, 1)
#for wine
#df_test3['cluster'] = np.where(df_test3['score']<4.75, 0, 1)

#varying (pima: 2.5, ionosphere: 6)
#df_test3['cluster'] = np.where(df_test3['score']<3, 0, 1)

df_test3['cluster'].value_counts()

df_test3.groupby('cluster').mean()
print(df_test3['cluster'].value_counts())

#predictions by threshold
y_pred = df_test3['cluster']

predplot(y_pred)
actualplot()

from pyod.models.combination import aom, moa, average, maximization

# Put all the predictions in a data frame
train_scores = pd.DataFrame({'clf1': clf1.decision_scores_,
                             'clf2': clf2.decision_scores_,
                             'clf3': clf3.decision_scores_
                            })



test_scores  = pd.DataFrame({'clf1': clf1.decision_function(scaled),
                             'clf2': clf2.decision_function(scaled),
                             'clf3': clf3.decision_function(scaled) 
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

df_test = pd.DataFrame(scaled)
df_test['y_by_average_score'] = y_by_average
#for lympho
df_test['y_by_average_cluster'] = np.where(df_test['y_by_average_score']<2, 0, 1)

#for glass
#df_test['y_by_average_cluster'] = np.where(df_test['y_by_average_score']<2, 0, 1)

#for wine
#df_test['y_by_average_cluster'] = np.where(df_test['y_by_average_score']<2, 0, 1)

#varying (pima: 0.5, ionosphere: 0.25)
#df_test['y_by_average_cluster'] = np.where(df_test['y_by_average_score']<1, 0, 1)

print(df_test['y_by_average_cluster'].value_counts())

y_pred = df_test['y_by_average_cluster']

#summary statistics
df_test.groupby('y_by_average_cluster').mean()

#plot
predplot(y_pred)
actualplot()


#%%