#%%
import scipy.io
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.hbos import HBOS
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def modelEvals(a_out, y_pred):
    #potential confusion matrix
    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
    y_true = a_out
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
    tpr = tp / actyes
    fpr = fp / actno
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

#run HBOS

nbins = "auto"
hbos = HBOS(n_bins=nbins)
hbos.fit(x_pca)
HBOS(alpha=0.1,n_bins=nbins, tol=0.5)
        
#yp = hbos.predict(x_pca) 
        
#generate anomaly scores
y_scores = hbos.decision_function(x_pca)
     
#predict anomalies (0 and 1) 
        
predprob = pd.DataFrame(hbos.predict(x_pca, return_confidence=True))
predprob = predprob.T
predprob = predprob.rename(columns={0:'Prediction', 1:'Probability'})
y_pred = predprob['Prediction']

#predicted outlier counts
unique, counts = np.unique(y_pred, return_counts=True)
ucounts = dict(zip(unique, counts))
#ax = sns.countplot(x = y_pred)
#plt.show()

plt.hist(y_scores, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram of anomaly scores with 'auto' bins")
plt.show()

#dataframe of predictions
pred_df = pd.DataFrame()
#pred_df['prediction'] = y_pred
pred_df['score'] = y_scores
pred_df['probability'] = predprob['Probability']
pred_df['prediction'] = y_pred
predval = pred_df['prediction']


#print number of Outliers and Inliers
out_in = predval.value_counts()
print("Outlier Counts: 0: inliers 1: outliers")
print(out_in)

        
print("\n")


        
#plot with highlighted outliers
print("Plot of Predicted Outliers:")
sns.scatterplot(data = x_pca, x = 'PC1', y = 'PC2', hue = predval)
plt.show()




def actualplot(a_out):
    print("Outlier Counts: 0: inliers 1: outliers")
    print(a_out.value_counts())
    print("plot of actual outliers")
    sns.scatterplot(data = x_pca, x = 'PC1', y = 'PC2', hue = a_out)
    plt.show()


#plot actual outliers
actualplot(a_out)

#y_pred = 
#evaluations
modelEvals(a_out,y_pred2)







# %%
