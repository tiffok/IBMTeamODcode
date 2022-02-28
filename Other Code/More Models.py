#%%
import scipy.io
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from pyod.models.mcd import MCD
from pyod.models.rod import ROD


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
udata = scipy.io.loadmat("wine.mat")


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

#run largest knn
print("KNN largest method")
lknn = KNN(n_neighbors=20, method='largest')

#fit onto data
lknn.fit(x_pca)

#generate anomaly scores
y_scores = lknn.decision_function(x_pca)

#generate predictions
y_preds = lknn.fit_predict(x_pca)

df = pd.DataFrame()
df['scores'] = y_scores
df['predictions'] =y_preds

y_pred = pd.Series(y_preds)
print("Predicted Outlier Counts: 0: inliers 1: outliers")
print(y_pred.value_counts())

predplot(y_pred)
actualplot(a_out)
modelEvals(a_out,y_pred)

#run Average KNN
print("Average KNN method")
aknn = KNN(n_neighbors=20,method='mean')

#fit onto data
aknn.fit(x_pca)

#generate anomaly scores
y_scores2 = aknn.decision_function(x_pca)

#generate predictions
y_preds2 = aknn.fit_predict(x_pca)

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

#run Median KNN
print("Median KNN method")
mknn = KNN(n_neighbors=20, method='median')

#fit onto data
mknn.fit(x_pca)

#generate anomaly scores
y_scores3 = mknn.decision_function(x_pca)

#generate predictions
y_preds3 = mknn.fit_predict(x_pca)

df3 = pd.DataFrame()
df3['scores'] = y_scores3
df3['predictions'] =y_preds3

y_pred3 = pd.Series(y_preds3)
print("Predicted Outlier Counts: 0: inliers 1: outliers")
print(y_pred3.value_counts())

#plots of the predicted and actual outliers, run assessments/ confusions matrix
predplot(y_pred3)
actualplot(a_out)
modelEvals(a_out,y_pred3)

#run OCSVM
print("OCSVM method")
ocsvm = OCSVM()

#fit onto data
ocsvm.fit(x_pca)

#generate anomaly scores
y_scores4 = ocsvm.decision_function(x_pca)

#generate predictions
y_preds4 = ocsvm.fit_predict(x_pca)

df4 = pd.DataFrame()
df4['scores'] = y_scores4
df4['predictions'] =y_preds4

y_pred4 = pd.Series(y_preds4)
print("Predicted Outlier Counts: 0: inliers 1: outliers")
print(y_pred4.value_counts())

#plots of the predicted and actual outliers, run assessments/ confusions matrix
predplot(y_pred4)
actualplot(a_out)
modelEvals(a_out,y_pred4)

#run MCD
print("Minimum Covariance Determinant method")
mcd = MCD()

#fit onto data
mcd.fit(x_pca)

#generate anomaly scores
y_scores5 = mcd.decision_function(x_pca)

#generate predictions
y_preds5 = mcd.fit_predict(x_pca)

df5 = pd.DataFrame()
df5['scores'] = y_scores5
df5['predictions'] =y_preds5

y_pred5 = pd.Series(y_preds5)
print("Predicted Outlier Counts: 0: inliers 1: outliers")
print(y_pred5.value_counts())

#plots of the predicted and actual outliers, run assessments/ confusions matrix
predplot(y_pred5)
actualplot(a_out)
modelEvals(a_out,y_pred5)

#run rotational OD
print("ROD method")
rod = ROD()

#fit onto data
rod.fit(x_pca)

#generate anomaly scores
y_scores6 = rod.decision_function(x_pca)

#generate predictions
y_preds6 = rod.fit_predict(x_pca)

df6 = pd.DataFrame()
df6['scores'] = y_scores6
df6['predictions'] =y_preds6

y_pred6 = pd.Series(y_preds6)
print("Predicted Outlier Counts: 0: inliers 1: outliers")
print(y_pred6.value_counts())

#plots of the predicted and actual outliers, run assessments/ confusions matrix
predplot(y_pred6)
actualplot(a_out)
modelEvals(a_out,y_pred6)



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
#fpr7, tpr7, threshold7 = roc_curve(a_out, y_pred7)
plt.plot(fpr1, tpr1, color = "darkorange", label = "KNN Largest Method Model")
plt.plot(fpr2, tpr2, color = "red", label = "KNN Mean Model")
plt.plot(fpr3, tpr3, color = "green", label = "KNN Median Model")
plt.plot(fpr4, tpr4, color = "lightblue", label = "OCSVM Model")
plt.plot(fpr5, tpr5, color = "purple", label = "MCD Model")
plt.plot(fpr6, tpr6, color = "deeppink", label = "ROD Model")
#plt.plot(fpr7, tpr7, color = "greenyellow", label = "ECOD Model")
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()

# %%
