#%%
from bokeh.plotting import figure, show
import panel as pn
from bokeh.models import ColumnDataSource, HoverTool, WheelZoomTool, ResetTool, PanTool
from bokeh.transform import factor_cmap
from bokeh.palettes import Category10

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
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.pca import PCA as PCA2

def actualplot(a_out):
    print("Actual Outlier Counts: 0: inliers 1: outliers")
    print(a_out.value_counts())
    print("plot of actual outliers")
    sns.scatterplot(data = x_pca, x = 'PC1', y = 'PC2', hue = a_out)
    plt.show()


def predplot(df, name):
    source = ColumnDataSource(df)
    hover = HoverTool(
        tooltips = [
            ("Anomaly Score", "@scores"),
            ("Outlier Classification", "@predout"),
            ("Outlier Classification Status", "@class_status")
        ]
    )
    
    cfactor = df['predout'].unique()
    
    npal = ('#ff7f0e','#1f77b4')
    rpal = ('#1f77b4','#ff7f0e')
    p = figure(tools = [hover, WheelZoomTool(), ResetTool(), PanTool()], title='Plot of Predicted Outliers for {}'.format(name))
    p.scatter(
        'PC1', 'PC2', source = source, 
        color = factor_cmap('predout', palette=npal, factors=cfactor),
        legend_group = 'predout')
    show(p)

def modelEvals(a_out, y_pred, y_score, df):
    #confusion matrix
    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay
    y_true = a_out
    
    #display confusion matrix
    c_mat = confusion_matrix(y_true, y_pred, labels=[0,1])
    c_disp = ConfusionMatrixDisplay(confusion_matrix=c_mat, display_labels=[0,1])
    c_disp.plot(cmap="bone_r")
    plt.show()


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
    auc = roc_auc_score(a_out,y_score)

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
    probab = df['probabilities']
    nfpr, ntpr, threshold = roc_curve(a_out, probab, pos_label=0)
    plt.plot(nfpr, ntpr, color = "darkorange", label = f"Model: AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Outlier Detection Method")
    plt.legend(loc="lower right")
    plt.show()

#read in data
#read in mat file
udata = scipy.io.loadmat("thyroid.mat")


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

#set contamination level
contam = 0.1

#ABOD
print("ABOD method")
name = 'ABOD'
#create the model
clf = ABOD(contamination=contam)

#fit the data
clf.fit(x_pca)

#raw outlier scores
y_scores = clf.decision_function(x_pca)

#generate predictions
y_preds = clf.fit_predict(x_pca)

#generate probabilities
proba = clf.predict_proba(x_pca)

df1 = pd.DataFrame(x_pca.copy())
df1['scores'] = y_scores
df1['predictions'] =y_preds
df1['truth'] = a_out
df1['predout'] = np.where(df1['predictions']==1,"Outlier","Inlier")
df1['trueout'] = np.where(df1['truth']==1,"Outlier","Inlier")
df1['probabilities'] = proba[:,0]

#create column for correct/incorrectly classified outliers

#sum predictions and actual classifications
fn = lambda row: row.predictions + row.truth
df1['c'] = df1.apply(fn, axis = 1)

#possible conditions
conditions = [
(df1['c'] == 2),
(df1['c'] == 1) & (df1['predictions'] == 1),
(df1['c'] == 1) & (df1['predictions'] == 0),
(df1['c'] ==0)]

class_values = [
    'Correctly Classified Outlier', 
    'Incorrectly Classified Outlier (Actual Inlier)', 
    'Incorrectly Classified Inlier (Actual Outlier)', 
    'Correctly Classified Inlier']

df1['class_status'] = np.select(conditions, class_values)

y_pred = pd.Series(y_preds)
print("Predicted Outlier Counts: 0: inliers 1: outliers")
print(y_pred.value_counts())

predplot(df1, name)
#actualplot(a_out)
modelEvals(a_out,y_pred, y_scores, df1)

#run CBLOF
print("CBLOF method")
name = 'CBLOF'
clf = CBLOF(contamination=contam)

#fit onto data
clf.fit(x_pca)

#generate anomaly scores
y_scores2 = clf.decision_function(x_pca)

#generate predictions
y_preds2 = clf.fit_predict(x_pca)

#generate probabilities
proba2 = clf.predict_proba(x_pca)

df2 = pd.DataFrame(x_pca.copy())
df2['scores'] = y_scores2
df2['predictions'] =y_preds2
df2['truth'] = a_out
df2['predout'] = np.where(df2['predictions']==1,"Outlier","Inlier")
df2['trueout'] = np.where(df2['truth']==1,"Outlier","Inlier")
df2['probabilities'] = proba2[:,0]

#create column for correct/incorrectly classified outliers

#sum predictions and actual classifications
fn = lambda row: row.predictions + row.truth
df2['c'] = df2.apply(fn, axis = 1)

#possible conditions
conditions = [
(df2['c'] == 2),
(df2['c'] == 1) & (df2['predictions'] == 1),
(df2['c'] == 1) & (df2['predictions'] == 0),
(df2['c'] ==0)]

class_values = [
    'Correctly Classified Outlier', 
    'Incorrectly Classified Outlier (Actual Inlier)', 
    'Incorrectly Classified Inlier (Actual Outlier)', 
    'Correctly Classified Inlier']

df2['class_status'] = np.select(conditions, class_values)

y_pred2 = pd.Series(y_preds2)
print("Predicted Outlier Counts: 0: inliers 1: outliers")
print(y_pred2.value_counts())

#plots of the predicted and actual outliers, run assessments/ confusions matrix
predplot(df2, name)
#actualplot(a_out)
modelEvals(a_out,y_pred2, y_scores2,df2)

#run Feature Bagging
print("Feature Bagging method")
name = 'Feature Bagging'
clf = FeatureBagging(contamination=contam)

#fit onto data
clf.fit(x_pca)

#generate anomaly scores
y_scores3 = clf.decision_function(x_pca)

#generate predictions
y_preds3 = clf.fit_predict(x_pca)

#generate probabilities
proba3 = clf.predict_proba(x_pca)

df3 = pd.DataFrame(x_pca.copy())
df3['scores'] = y_scores3
df3['predictions'] =y_preds3
df3['truth'] = a_out
df3['predout'] = np.where(df3['predictions']==1,"Outlier","Inlier")
df3['trueout'] = np.where(df3['truth']==1,"Outlier","Inlier")
df3['probabilities'] = proba3[:,0]

#create column for correct/incorrectly classified outliers

#sum predictions and actual classifications
fn = lambda row: row.predictions + row.truth
df3['c'] = df3.apply(fn, axis = 1)

#possible conditions
conditions = [
(df3['c'] == 2),
(df3['c'] == 1) & (df3['predictions'] == 1),
(df3['c'] == 1) & (df3['predictions'] == 0),
(df3['c'] ==0)]

class_values = [
    'Correctly Classified Outlier', 
    'Incorrectly Classified Outlier (Actual Inlier)', 
    'Incorrectly Classified Inlier (Actual Outlier)', 
    'Correctly Classified Inlier']

df3['class_status'] = np.select(conditions, class_values)

y_pred3 = pd.Series(y_preds3)
print("Predicted Outlier Counts: 0: inliers 1: outliers")
print(y_pred3.value_counts())

#plots of the predicted and actual outliers, run assessments/ confusions matrix
predplot(df3, name)
#actualplot(a_out)
modelEvals(a_out,y_pred3, y_scores3,df3)

#run HBOS
print("HBOS method")
name = 'HBOS'
clf = HBOS(n_bins = 'auto', contamination=contam)

#fit onto data
clf.fit(x_pca)

#generate anomaly scores
y_scores4 = clf.decision_function(x_pca)

#generate predictions
y_preds4 = clf.fit_predict(x_pca)

#generate probabilities
proba4 = clf.predict_proba(x_pca)

df4 = pd.DataFrame(x_pca.copy())
df4['scores'] = y_scores4
df4['predictions'] =y_preds4
df4['truth'] = a_out
df4['predout'] = np.where(df4['predictions']==1,"Outlier","Inlier")
df4['trueout'] = np.where(df4['truth']==1,"Outlier","Inlier")
df4['probabilities'] = proba4[:,0]

#create column for correct/incorrectly classified outliers

#sum predictions and actual classifications
fn = lambda row: row.predictions + row.truth
df4['c'] = df4.apply(fn, axis = 1)

#possible conditions
conditions = [
(df4['c'] == 2),
(df4['c'] == 1) & (df4['predictions'] == 1),
(df4['c'] == 1) & (df4['predictions'] == 0),
(df4['c'] ==0)]

class_values = [
    'Correctly Classified Outlier', 
    'Incorrectly Classified Outlier (Actual Inlier)', 
    'Incorrectly Classified Inlier (Actual Outlier)', 
    'Correctly Classified Inlier']

df4['class_status'] = np.select(conditions, class_values)

y_pred4 = pd.Series(y_preds2)
print("Predicted Outlier Counts: 0: inliers 1: outliers")
print(y_pred4.value_counts())

#plots of the predicted and actual outliers, run assessments/ confusions matrix
predplot(df4, name)
#actualplot(a_out)
modelEvals(a_out,y_pred4, y_scores4,df4)

#run Iforest
print("Iforest method")
name = 'IForest'
clf = IForest(contamination=contam)

#fit onto data
clf.fit(x_pca)

#generate anomaly scores
y_scores5 = clf.decision_function(x_pca)

#generate predictions
y_preds5 = clf.fit_predict(x_pca)

#generate probabilities
proba5 = clf.predict_proba(x_pca)

df5 = pd.DataFrame(x_pca.copy())
df5['scores'] = y_scores5
df5['predictions'] =y_preds5
df5['truth'] = a_out
df5['predout'] = np.where(df5['predictions']==1,"Outlier","Inlier")
df5['trueout'] = np.where(df5['truth']==1,"Outlier","Inlier")
df5['probabilities'] = proba5[:,0]

#create column for correct/incorrectly classified outliers

#sum predictions and actual classifications
fn = lambda row: row.predictions + row.truth
df5['c'] = df5.apply(fn, axis = 1)

#possible conditions
conditions = [
(df5['c'] == 2),
(df5['c'] == 1) & (df5['predictions'] == 1),
(df5['c'] == 1) & (df5['predictions'] == 0),
(df5['c'] ==0)]

class_values = [
    'Correctly Classified Outlier', 
    'Incorrectly Classified Outlier (Actual Inlier)', 
    'Incorrectly Classified Inlier (Actual Outlier)', 
    'Correctly Classified Inlier']

df5['class_status'] = np.select(conditions, class_values)

y_pred5 = pd.Series(y_preds5)
print("Predicted Outlier Counts: 0: inliers 1: outliers")
print(y_pred5.value_counts())

#plots of the predicted and actual outliers, run assessments/ confusions matrix
predplot(df5, name)
#actualplot(a_out)
modelEvals(a_out,y_pred5, y_scores5,df5)

#run largest knn
print("KNN largest method")
name = 'KNN'
clf = KNN(n_neighbors=20, method='largest', contamination=contam)

#fit onto data
clf.fit(x_pca)

#generate anomaly scores
y_scores6 = clf.decision_function(x_pca)

#generate predictions
y_preds6 = clf.fit_predict(x_pca)

#generate probabilities
proba6 = clf.predict_proba(x_pca)

df6 = pd.DataFrame(x_pca.copy())
df6['scores'] = y_scores6
df6['predictions'] =y_preds6
df6['truth'] = a_out
df6['predout'] = np.where(df6['predictions']==1,"Outlier","Inlier")
df6['trueout'] = np.where(df6['truth']==1,"Outlier","Inlier")
df6['probabilities'] = proba6[:,0]

#create column for correct/incorrectly classified outliers

#sum predictions and actual classifications
fn = lambda row: row.predictions + row.truth
df6['c'] = df6.apply(fn, axis = 1)

#possible conditions
conditions = [
(df6['c'] == 2),
(df6['c'] == 1) & (df6['predictions'] == 1),
(df6['c'] == 1) & (df6['predictions'] == 0),
(df6['c'] ==0)]

class_values = [
    'Correctly Classified Outlier', 
    'Incorrectly Classified Outlier (Actual Inlier)', 
    'Incorrectly Classified Inlier (Actual Outlier)', 
    'Correctly Classified Inlier']

df6['class_status'] = np.select(conditions, class_values)


y_pred6 = pd.Series(y_preds6)
print("Predicted Outlier Counts: 0: inliers 1: outliers")
print(y_pred6.value_counts())

#plots of the predicted and actual outliers, run assessments/ confusions matrix
predplot(df6, name)
#actualplot(a_out)
modelEvals(a_out,y_pred6, y_scores6,df6)

#run LOF
print("LOF method")
name = 'LOF'
clf = LOF(algorithm='auto', contamination=contam)

#fit onto data
clf.fit(x_pca)

#generate anomaly scores
y_scores7 = clf.decision_function(x_pca)

#generate predictions
y_preds7 = clf.fit_predict(x_pca)

#generate probabilities
proba7 = clf.predict_proba(x_pca)

df7 = pd.DataFrame(x_pca.copy())
df7['scores'] = y_scores7
df7['predictions'] =y_preds7
df7['truth'] = a_out
df7['predout'] = np.where(df7['predictions']==1,"Outlier","Inlier")
df7['trueout'] = np.where(df7['truth']==1,"Outlier","Inlier")
df7['probabilities'] = proba7[:,0]

#create column for correct/incorrectly classified outliers

#sum predictions and actual classifications
fn = lambda row: row.predictions + row.truth
df7['c'] = df7.apply(fn, axis = 1)

#possible conditions
conditions = [
(df7['c'] == 2),
(df7['c'] == 1) & (df7['predictions'] == 1),
(df7['c'] == 1) & (df7['predictions'] == 0),
(df7['c'] ==0)]

class_values = [
    'Correctly Classified Outlier', 
    'Incorrectly Classified Outlier (Actual Inlier)', 
    'Incorrectly Classified Inlier (Actual Outlier)', 
    'Correctly Classified Inlier']

df7['class_status'] = np.select(conditions, class_values)


y_pred7 = pd.Series(y_preds7)
print("Predicted Outlier Counts: 0: inliers 1: outliers")
print(y_pred7.value_counts())

#plots of the predicted and actual outliers, run assessments/ confusions matrix
predplot(df7, name)
#actualplot(a_out)
modelEvals(a_out,y_pred7, y_scores7,df7)

#run MCD
print("Minimum Covariance Determinant method")
name = 'MCD'
clf = MCD(contamination=contam)

#fit onto data
clf.fit(x_pca)

#generate anomaly scores
y_scores8 = clf.decision_function(x_pca)

#generate predictions
y_preds8 = clf.fit_predict(x_pca)

#generate probabilities
proba8 = clf.predict_proba(x_pca)

df8 = pd.DataFrame(x_pca.copy())
df8['scores'] = y_scores8
df8['predictions'] =y_preds8
df8['truth'] = a_out
df8['predout'] = np.where(df8['predictions']==1,"Outlier","Inlier")
df8['trueout'] = np.where(df8['truth']==1,"Outlier","Inlier")
df8['probabilities'] = proba8[:,0]

#create column for correct/incorrectly classified outliers

#sum predictions and actual classifications
fn = lambda row: row.predictions + row.truth
df8['c'] = df8.apply(fn, axis = 1)

#possible conditions
conditions = [
(df8['c'] == 2),
(df8['c'] == 1) & (df8['predictions'] == 1),
(df8['c'] == 1) & (df8['predictions'] == 0),
(df8['c'] ==0)]

class_values = [
    'Correctly Classified Outlier', 
    'Incorrectly Classified Outlier (Actual Inlier)', 
    'Incorrectly Classified Inlier (Actual Outlier)', 
    'Correctly Classified Inlier']

df8['class_status'] = np.select(conditions, class_values)
y_pred8 = pd.Series(y_preds8)
print("Predicted Outlier Counts: 0: inliers 1: outliers")
print(y_pred8.value_counts())

#plots of the predicted and actual outliers, run assessments/ confusions matrix
predplot(df8, name)
#actualplot(a_out)
modelEvals(a_out,y_pred8, y_scores8,df8)

#run OCSVM
print("OCSVM method")
name = 'OCSVM'
clf = OCSVM(contamination=contam)

#fit onto data
clf.fit(x_pca)

#generate anomaly scores
y_scores9 = clf.decision_function(x_pca)

#generate predictions
y_preds9 = clf.fit_predict(x_pca)

#generate probabilities
proba9 = clf.predict_proba(x_pca)

df9 = pd.DataFrame(x_pca.copy())
df9['scores'] = y_scores9
df9['predictions'] =y_preds9
df9['truth'] = a_out
df9['predout'] = np.where(df9['predictions']==1,"Outlier","Inlier")
df9['trueout'] = np.where(df9['truth']==1,"Outlier","Inlier")
df9['probabilities'] = proba9[:,0]

#create column for correct/incorrectly classified outliers

#sum predictions and actual classifications
fn = lambda row: row.predictions + row.truth
df9['c'] = df9.apply(fn, axis = 1)

#possible conditions
conditions = [
(df9['c'] == 2),
(df9['c'] == 1) & (df9['predictions'] == 1),
(df9['c'] == 1) & (df9['predictions'] == 0),
(df9['c'] ==0)]

class_values = [
    'Correctly Classified Outlier', 
    'Incorrectly Classified Outlier (Actual Inlier)', 
    'Incorrectly Classified Inlier (Actual Outlier)', 
    'Correctly Classified Inlier']

df9['class_status'] = np.select(conditions, class_values)


y_pred9 = pd.Series(y_preds9)
print("Predicted Outlier Counts: 0: inliers 1: outliers")
print(y_pred9.value_counts())

#plots of the predicted and actual outliers, run assessments/ confusions matrix
predplot(df9, name)
#actualplot(a_out)
modelEvals(a_out,y_pred9, y_scores9,df9)


#run PCA
print("PCA method")
name = 'PCA'
clf = PCA2(contamination=contam)

#fit onto data
clf.fit(x_pca)

#generate anomaly scores
y_scores10 = clf.decision_function(x_pca)

#generate predictions
y_preds10 = clf.fit_predict(x_pca)

#generate probabilities
proba10 = clf.predict_proba(x_pca)

df10 = pd.DataFrame(x_pca.copy())
df10['scores'] = y_scores10
df10['predictions'] =y_preds10
df10['truth'] = a_out
df10['predout'] = np.where(df10['predictions']==1,"Outlier","Inlier")
df10['trueout'] = np.where(df10['truth']==1,"Outlier","Inlier")
df10['probabilities'] = proba10[:,0]

#create column for correct/incorrectly classified outliers

#sum predictions and actual classifications
fn = lambda row: row.predictions + row.truth
df10['c'] = df10.apply(fn, axis = 1)

#possible conditions
conditions = [
(df10['c'] == 2),
(df10['c'] == 1) & (df10['predictions'] == 1),
(df10['c'] == 1) & (df10['predictions'] == 0),
(df10['c'] ==0)]

class_values = [
    'Correctly Classified Outlier', 
    'Incorrectly Classified Outlier (Actual Inlier)', 
    'Incorrectly Classified Inlier (Actual Outlier)', 
    'Correctly Classified Inlier']

df10['class_status'] = np.select(conditions, class_values)


y_pred10 = pd.Series(y_preds10)
print("Predicted Outlier Counts: 0: inliers 1: outliers")
print(y_pred6.value_counts())

#plots of the predicted and actual outliers, run assessments/ confusions matrix
predplot(df10, name)
#actualplot(a_out)
modelEvals(a_out,y_pred10, y_scores10,df10)




# %%
#multi ROC Curve
from sklearn.metrics import roc_curve, roc_auc_score
print("Multi ROC curve")
fpr1, tpr1, threshold = roc_curve(a_out, proba[:,0], pos_label=0)
auc1 = roc_auc_score(a_out, y_scores)

fpr2, tpr2, threshold2 = roc_curve(a_out, proba2[:,0], pos_label=0)
auc2 = roc_auc_score(a_out, y_scores2)

fpr3, tpr3, threshold3 = roc_curve(a_out, proba3[:,0], pos_label=0)
auc3 = roc_auc_score(a_out, y_scores3)

fpr4, tpr4, threshold4 = roc_curve(a_out, proba4[:,0], pos_label=0)
auc4 = roc_auc_score(a_out, y_scores4)

fpr5, tpr5, threshold5 = roc_curve(a_out, proba5[:,0], pos_label=0)
auc5 = roc_auc_score(a_out, y_scores5)

fpr6, tpr6, threshold6 = roc_curve(a_out,proba6[:,0],  pos_label=0)
auc6 = roc_auc_score(a_out, y_scores6)

fpr7, tpr7, threshold7 = roc_curve(a_out, proba7[:,0], pos_label=0)
auc7 = roc_auc_score(a_out, y_scores7)

fpr8, tpr8, threshold8 = roc_curve(a_out, proba8[:,0], pos_label=0)
auc8 = roc_auc_score(a_out, y_scores8)

fpr9, tpr9, threshold9 = roc_curve(a_out, proba9[:,0], pos_label=0)
auc9 = roc_auc_score(a_out, y_scores9)

fpr10, tpr10, threshold10 = roc_curve(a_out, proba10[:,0], pos_label=0)
auc10 = roc_auc_score(a_out, y_scores10)

plt.plot(fpr1, tpr1, color = "darkorange", label = f"ABOD Model: AUC = {auc1:.3f}")
plt.plot(fpr2, tpr2, color = "red", label = f"CBLOF Model: AUC = {auc2:.3f} ")
plt.plot(fpr3, tpr3, color = "green", label = f"FB Model: AUC = {auc3:.3f}")
plt.plot(fpr4, tpr4, color = "lightblue", label = f"HBOS Model: AUC = {auc4:.3f}")
plt.plot(fpr5, tpr5, color = "purple", label = f"IForest Model: AUC = {auc5:.3f}")
plt.plot(fpr6, tpr6, color = "deeppink", label = f"KNN Model: AUC = {auc6:.3f}")
plt.plot(fpr7, tpr7, color = "greenyellow", label = f"LOF Model: AUC = {auc7:.3f}")
plt.plot(fpr8, tpr8, color = "mediumvioletred", label = f"MCD Model: AUC = {auc8:.3f}")
plt.plot(fpr9, tpr9, color = "indigo", label = f"OCSVM Model: AUC = {auc9:.3f}")
plt.plot(fpr10, tpr10, color = "slategray", label = f"PCA Model: AUC = {auc10:.3f}")
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Outlier Detection Method")
plt.legend()
plt.show()

# %%
