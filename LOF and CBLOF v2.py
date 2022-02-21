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
from pyod.models.cblof import CBLOF
from pyod.models.lof import LOF

def actualplot(a_out):
    print("Actual Outlier Counts: 0: inliers 1: outliers")
    print(a_out.value_counts())
    print("plot of actual outliers")
    sns.scatterplot(data = x_pca, x = 'PC1', y = 'PC2', hue = a_out)
    plt.show()

def predplot(y_pred):
    print("plot of predicted outliers")
    sns.scatterplot(data = x_pca, x = 'PC1', y = 'PC2', hue = y_pred)
    plt.show()

def modelEvals(a_out, y_pred):
    #confusion matrix
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

#read in data
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

#run PCA
pca = PCA(2)
x_pca = pca.fit_transform(scaled)
x_pca = pd.DataFrame(x_pca)
x_pca.columns=['PC1','PC2']

#run LOF
lof = LOF(algorithm='auto')

#fit onto data
lof.fit(x_pca)

#generate anomaly scores
y_scores = lof.decision_function(x_pca)

#generate predictions
y_preds = lof.fit_predict(x_pca)

df = pd.DataFrame()
df['scores'] = y_scores
df['predictions'] =y_preds

y_pred = pd.Series(y_preds)
print("Predicted Outlier Counts: 0: inliers 1: outliers")
print(y_pred.value_counts())

predplot(y_pred)
actualplot(a_out)
modelEvals(a_out,y_pred)

#run CBLOF
cblof = CBLOF()

#fit onto data
cblof.fit(x_pca)

#generate anomaly scores
y_scores2 = cblof.decision_function(x_pca)

#generate predictions
y_preds2 = cblof.fit_predict(x_pca)

df2 = pd.DataFrame()
df2['scores'] = y_scores2
df2['predictions'] =y_preds2

y_pred2 = pd.Series(y_preds2)
print("Predicted Outlier Counts: 0: inliers 1: outliers")
print(y_pred.value_counts())

#plots of the predicted and actual outliers, run assessments/ confusions matrix
predplot(y_pred2)
actualplot(a_out)
modelEvals(a_out,y_pred2)













# %%
#multi ROC Curve
from sklearn.metrics import roc_curve
print("Multi ROC curve")
nfpr, ntpr, threshold = roc_curve(a_out, y_pred)
mfpr, mtpr, threshold2 = roc_curve(a_out, y_pred2)
plt.plot(nfpr, ntpr, color = "darkorange", label = "LOF Model")
plt.plot(mfpr, mtpr, color = "red", label = "CBLOF Model")
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()


# %%
