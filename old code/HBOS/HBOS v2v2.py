""""
author: Tiffany Okorochukwu
"""
#medium article: https://medium.com/dataman-in-ai/anomaly-detection-with-histogram-based-outlier-detection-hbo-bc10ef52f23f

#load in some packages
from enum import auto
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.hbos import HBOS
from sklearn.preprocessing import MinMaxScaler

#running the HBOS function
def runHBOS(X):
        #print the column names
        print("For attributes: " + str(X.columns[0]) + " and " + str(X.columns[1]))
        
        #scale data
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(X)
        scaled = pd.DataFrame(scaled, columns=X.columns)

        #run HBOS

        nbins = 50
        hbos = HBOS(n_bins=nbins)
        hbos.fit(scaled)
        HBOS(alpha=0.1,n_bins=nbins, tol=0.5)
        #generate anomaly scores
        y_scores = hbos.decision_function(scaled)
        
        #predict anomalies (0 and 1) 
        
        predprob = pd.DataFrame(hbos.predict(scaled, return_confidence=True))
        predprob = predprob.T
        predprob = predprob.rename(columns={0:'Prediction', 1:'Probability'})
        y_pred = predprob['Prediction']

        #predicted outlier counts
        unique, counts = np.unique(y_pred, return_counts=True)
        ucounts = dict(zip(unique, counts))
        #ax = sns.countplot(x = y_pred)
        #plt.show()

        #dataframe of predictions
        pred_df = pd.DataFrame(scaled)
        #pred_df['prediction'] = y_pred
        pred_df['score'] = y_scores
        pred_df['probability'] = predprob['Probability']
        pred_df['prediction'] = np.where(y_pred<1, 0, 1)
        #pred_df['actual'] = a_out
        #pred_df['tots'] = pred_df.iloc[:,4:6].sum(axis=1)

        #print number of Outliers and Inliers
        out_in = pred_df['prediction'].value_counts()
        print("Predicted " + str(out_in[0]) + " inliers and " + str(out_in[1]) + " outliers.")
        print("\n")


        # summary statistics:
        print(pred_df.groupby('prediction').mean())
        print("\n")

        
        #plot with highlighted outliers
        sns.scatterplot(data = scaled, x = scaled.columns[0], y = scaled.columns[1], hue = y_pred)
        plt.show()

#read in the data
data = pd.read_csv("https://raw.githubusercontent.com/DaviRolim/UCI-Wine-DataSet/master/wine.csv")

#wine data without headers
wine = data.iloc[:,1:14]

#show first rows
print(wine.head())

l_range = len(wine.columns) -1


#create actual outliers column
y = data.iloc[:,0]
c_df = pd.DataFrame()
c_df['Class'] = y
c_df['actual'] = np.where(c_df['Class']==1,1,0)

a_out = c_df['actual']

for i in range(len(wine.columns)) :
    if (i == l_range) :
        #select data
        X = wine.iloc[:,[l_range,0]]
        runHBOS(X)
        
        break
    X = wine.iloc[:,i:i+2]
    runHBOS(X)
      



