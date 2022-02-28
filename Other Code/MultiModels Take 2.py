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
from pyod.models.loda import LODA
from pyod.models.abod import ABOD
from pyod.models.loci import LOCI
from pyod.models.cof import COF
from pyod.models.ecod import ECOD

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
udata = scipy.io.loadmat("cardio.mat")


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
print("LOF method")
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
print("CBLOF method")
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
print(y_pred2.value_counts())

#plots of the predicted and actual outliers, run assessments/ confusions matrix
predplot(y_pred2)
actualplot(a_out)
modelEvals(a_out,y_pred2)

#LODA
print("LODA method")
#create the model
loda = LOCI()

#fit the data
loda.fit(x_pca)

#raw outlier scores
y_scores = loda.decision_function(x_pca)

#generate predictions
y_preds = loda.fit_predict(x_pca)

df = pd.DataFrame()
df['scores'] = y_scores
df['predictions'] =y_preds

y_pred3 = pd.Series(y_preds)
print("Predicted Outlier Counts: 0: inliers 1: outliers")
print(y_pred3.value_counts())

predplot(y_pred3)
actualplot(a_out)
modelEvals(a_out,y_pred3)

#ABOD
print("ABOD method")
#create the model
abod = ABOD()

#fit the data
abod.fit(x_pca)

#raw outlier scores
y_scores = abod.decision_function(x_pca)

#generate predictions
y_preds = abod.fit_predict(x_pca)

df = pd.DataFrame()
df['scores'] = y_scores
df['predictions'] =y_preds

y_pred4 = pd.Series(y_preds)
print("Predicted Outlier Counts: 0: inliers 1: outliers")
print(y_pred4.value_counts())

predplot(y_pred4)
actualplot(a_out)
modelEvals(a_out,y_pred4)

#LOCI
print("LOCI method")
#create the model
loci = LOCI()

#fit the data
loci.fit(x_pca)

#raw outlier scores
y_scores = loci.decision_function(x_pca)

#generate predictions
y_preds = loci.fit_predict(x_pca)

df = pd.DataFrame()
df['scores'] = y_scores
df['predictions'] =y_preds

y_pred5 = pd.Series(y_preds)
print("Predicted Outlier Counts: 0: inliers 1: outliers")
print(y_pred5.value_counts())

predplot(y_pred5)
actualplot(a_out)
modelEvals(a_out,y_pred5)


#COF
print("COF method")
#create the model
cof = COF()

#fit the data
cof.fit(x_pca)

#raw outlier scores
y_scores = cof.decision_function(x_pca)

#generate predictions
y_preds = cof.fit_predict(x_pca)

df = pd.DataFrame()
df['scores'] = y_scores
df['predictions'] =y_preds

y_pred6 = pd.Series(y_preds)
print("Predicted Outlier Counts: 0: inliers 1: outliers")
print(y_pred6.value_counts())

predplot(y_pred6)
actualplot(a_out)
modelEvals(a_out,y_pred6)

#ECOD
print("ECOD method")
#create the model
ecod = ECOD()

#fit the data
ecod.fit(x_pca)

#raw outlier scores
y_scores = ecod.decision_function(x_pca)

#generate predictions
y_preds = ecod.fit_predict(x_pca)

df = pd.DataFrame()
df['scores'] = y_scores
df['predictions'] =y_preds

y_pred7 = pd.Series(y_preds)
print("Predicted Outlier Counts: 0: inliers 1: outliers")
print(y_pred7.value_counts())

predplot(y_pred7)
actualplot(a_out)
modelEvals(a_out,y_pred7)


# %%
#multi ROC Curve
from sklearn.metrics import roc_curve
print("Multi ROC curve")
fpr1, tpr1, threshold = roc_curve(a_out, y_pred)
fpr2, tpr2, threshold2 = roc_curve(a_out, y_pred2)
fpr3, tpr3, threshold3 = roc_curve(a_out, y_pred3)
fpr4, tpr4, threshold4 = roc_curve(a_out, y_pred4)
fpr5, tpr5, threshold5 = roc_curve(a_out, y_pred5)
fpr6, tpr6, threshold6 = roc_curve(a_out, y_pred6)
fpr7, tpr7, threshold7 = roc_curve(a_out, y_pred7)
plt.plot(fpr1, tpr1, color = "darkorange", label = "LOF Model")
plt.plot(fpr2, tpr2, color = "red", label = "CBLOF Model")
plt.plot(fpr3, tpr3, color = "green", label = "LODA Model")
plt.plot(fpr4, tpr4, color = "lightblue", label = "ABOD Model")
plt.plot(fpr5, tpr5, color = "purple", label = "LOCI Model")
plt.plot(fpr6, tpr6, color = "deeppink", label = "COF Model")
plt.plot(fpr7, tpr7, color = "greenyellow", label = "ECOD Model")
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()


# %%
